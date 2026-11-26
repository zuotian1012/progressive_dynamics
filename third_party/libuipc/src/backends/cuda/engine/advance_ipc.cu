#include "flag_line_search.h"
#include <finite_element/constitutions/psq_preview_decision.h>
#include <limits>
#include <finite_element/finite_element_method.h> // PSQ adapter
#include <finite_element/constitutions/psq_proxy.h>
#include <cstdlib>
#include <sim_engine.h>
#include <uipc/common/range.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <line_search/line_searcher.h>
#include <linear_system/global_linear_system.h>
#include <animator/global_animator.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <newton_tolerance/newton_tolerance_manager.h>
#include <time_integrator/time_integrator_manager.h>

namespace uipc::backend::cuda
{
void SimEngine::advance()
{
    Float alpha     = 1.0;
    Float beta      = 1.0;
    Float ccd_alpha = 1.0;
    Float cfl_alpha = 1.0;
    const char* force_env = std::getenv("UIPC_PSQ_FORCE_UPDATE");
    const bool force_updates = force_env && std::string(force_env)=="1" && psq_replaces_bending();
    const char* inf_env=std::getenv("UIPC_PSQ_REJECT_INF_TRIAL");
    const bool reject_inf_trials=inf_env && std::string(inf_env)=="1" && psq_replaces_bending();
    bool forced_frame = false;
    const char* preview_env = std::getenv("UIPC_PSQ_PREVIEW_STOP");
    const bool preview_requested = preview_env && std::string(preview_env)=="1";
    if(preview_requested && force_updates)
        throw SimEngineException("PSQ preview and forced updates are mutually exclusive");

    /***************************************************************************************
    *                                  Function Shortcuts
    ***************************************************************************************/

    auto detect_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect DCD Candidates"};
            m_global_trajectory_filter->detect(0.0);
            m_global_trajectory_filter->filter_active();
        }
    };

    auto detect_trajectory_candidates = [this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Detect Trajectory Candidates"};
            m_global_trajectory_filter->detect(alpha);
        }
    };

    auto filter_dcd_candidates = [this]
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter Contact Candidates"};
            m_global_trajectory_filter->filter_active();
        }
    };

    auto record_friction_candidates = [this]
    {
        if(m_global_trajectory_filter && m_friction_enabled)
        {
            m_global_trajectory_filter->record_friction_candidates();
        }
    };

    auto compute_adaptive_kappa = [this]
    {
        if(m_global_contact_manager)
            m_global_contact_manager->compute_adaptive_parameters();
    };

    auto compute_dytopo_effect = [this]
    {
        // compute the dytopo effect gradient and hessian, containing:
        // 1) contact effect from contact pairs
        // 2) other dynamic topo effects, e.g. point picker, vertex stitch ...
        if(m_global_dytopo_effect_manager)
        {
            Timer timer{"Compute DyTopo Effect"};
            m_global_dytopo_effect_manager->compute_dytopo_effect();
        }
    };

    auto cfl_condition = [&cfl_alpha, this](Float alpha)
    {
        if(m_global_contact_manager)
        {
            Timer timer{"Compute CFL Condition"};
            cfl_alpha = m_global_contact_manager->compute_cfl_condition();
            m_frame_last_cfl_alpha = cfl_alpha;
            if(cfl_alpha < alpha)
            {
                logger::info("CFL Filter: {} < {}", cfl_alpha, alpha);
                return cfl_alpha;
            }
        }

        return alpha;
    };

    auto filter_toi = [&ccd_alpha, this](Float alpha)
    {
        if(m_global_trajectory_filter)
        {
            Timer timer{"Filter CCD TOI"};
            ccd_alpha = m_global_trajectory_filter->filter_toi(alpha);
            m_frame_last_ccd_toi = ccd_alpha;
            // ccd_alpha is a fraction of the swept step `alpha`; compose to
            // get the absolute step fraction
            Float absolute_alpha = alpha * ccd_alpha;
            if(absolute_alpha < alpha)
            {
                logger::info("CCD Filter: {} < {}", absolute_alpha, alpha);
                return absolute_alpha;
            }
        }

        return alpha;
    };

    // Stiff-GIPC design: cap the step with the feasible step over the active
    // contact set (each active pair keeps at least 20% of its current gap),
    // so the swept-AABB trajectory candidates generated afterwards are fewer.
    auto feasible_step = [this](Float alpha)
    {
        if(m_finite_element_method) alpha=psq_feasible_step(*m_finite_element_method,alpha,m_current_frame,m_newton_iter);
        if(m_global_contact_manager)
        {
            Timer timer{"Compute Feasible Step"};
            Float feasible_alpha = m_global_contact_manager->compute_feasible_step();
            if(feasible_alpha < alpha)
            {
                logger::info("Feasible Step Filter: {} < {}", feasible_alpha, alpha);
                return feasible_alpha;
            }
        }

        return alpha;
    };

    auto compute_energy = [this, filter_dcd_candidates](Float alpha) -> Float
    {
        // Step Forward => x = x_0 + alpha * dx
        m_global_vertex_manager->step_forward(alpha);
        m_line_searcher->step_forward(alpha);

        // Dumps surface after step forward
        if(m_dump_surface->view()[0])
        {
            dump_global_surface();
        }

        // Update the collision pairs
        filter_dcd_candidates();

        // Compute New Energy => E
        return m_line_searcher->compute_energy(false);
    };

    auto compute_animation_substep_ratio = [this](SizeT newton_iter)
    {
        // compute the ratio to the aim position.
        // dst = prev_position + ratio * (position - prev_position)
        if(m_global_animator)
        {
            m_global_animator->compute_substep_ratio(newton_iter);
            logger::info("Animation Substep Ratio: {}", m_global_animator->substep_ratio());
        }
    };

    auto animation_reach_target = [this]()
    {
        if(m_global_animator)
        {
            return m_global_animator->substep_ratio() >= 1.0;
        }
        return true;
    };

    auto convergence_check = [&](SizeT newton_iter) -> bool
    {
        if(!animation_reach_target())
            return false;

        if(m_semi_implicit_enabled)
        {
            // Ref: https://arxiv.org/abs/2512.12151, Algorithm 1.
            // newton/semi_implicit/K_min plays Stiff-GIPC's Kmin role: beta
            // accumulation only starts from this iteration on; it is not a
            // floor on the Newton iteration count (that is newton/min_iter,
            // default 0 = no forced minimum).
            auto k_min = m_semi_implicit_kmin->view()[0];
            auto eps   = m_semi_implicit_beta_tol;
            if(newton_iter >= k_min)
                beta = (1.0 - alpha) * beta;

            if(beta <= eps)
                return true;  // early terminate
        }

        NewtonToleranceManager::ResultInfo result_info;
        result_info.frame(m_current_frame);
        result_info.newton_iter(newton_iter);
        m_newton_tolerance_manager->check(result_info);

        if(!result_info.converged())
            return false;

        // ccd alpha should close to 1.0
        if(ccd_alpha < m_ccd_tol->view()[0])
            return false;

        return true;
    };

    auto update_diff_parm = [this]()
    {
        if(m_global_diff_sim_manager)
        {
            Timer timer{"Update Diff Parm"};
            m_global_diff_sim_manager->update();
        }
    };

    auto check_line_search_iter = [&](SizeT line_search_iter_after_loop, Float E0, Float last_E)
    {
        if(line_search_iter_after_loop >= m_line_searcher->max_iter())
        {
            m_frame_hit_line_search_limit = true;
            // alpha was halved once more after the last failed try,
            // so the last actually tried step length is 2*alpha
            Float alpha_last = 2.0 * alpha;
            Float abs_E0     = std::abs(E0);
            Float rel_E_increase = (last_E - E0) / (abs_E0 > 1e-30 ? abs_E0 : 1e-30);
            auto detail = fmt::format(
                "Line Search Exits with Max Iteration: {} (Frame={}, Newton={}, "
                "alpha_last={}, E0={:.6e}, E_last={:.6e}, rel_E_increase={:.3e}, "
                "ccd_alpha={}, cfl_alpha={})",
                m_line_searcher->max_iter(),
                m_current_frame,
                m_newton_iter,
                alpha_last,
                E0,
                last_E,
                rel_E_increase,
                ccd_alpha,
                cfl_alpha);
            logger::warn("{}", detail);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: " + detail);
            }
        }
    };

    auto check_newton_iter = [this, &forced_frame, force_updates](IndexT newton_iter_after_loop)
    {
        if(forced_frame || m_frame_preview_non_descent || m_frame_preview_small_step) return;
        auto newton_max = m_newton_max_iter->view()[0];
        if(newton_iter_after_loop >= newton_max)
        {
            m_frame_hit_newton_limit = true;
            if(force_updates)
            {
                forced_frame = true;
                m_frame_converged = false;
                logger::warn("PSQ_FORCE_ACCEPT frame={} reason=newton_limit newton={} alpha={:.17g}",
                             m_current_frame, newton_iter_after_loop-1, m_frame_last_line_search_alpha);
                return;
            }
            logger::warn("Newton Iteration Exits with Max Iteration: {} (Frame={})",
                         newton_max,
                         m_current_frame);

            if(m_strict_mode->view()[0])
            {
                throw SimEngineException("StrictMode: Newton Iteration Exits with Max Iteration");
            }
        }
        else
        {
            logger::info("Newton Iteration Converged with Iteration Count: {}, Max: {}",
                         newton_iter_after_loop,
                         newton_max);
        }
    };

    /***************************************************************************************
    *                                  Core Pipeline
    ***************************************************************************************/

    // Abort on exception if the runtime check is enabled for debugging
    constexpr bool AbortOnException = uipc::RUNTIME_CHECK;

    auto pipeline = [&]() noexcept(AbortOnException)
    {
        Timer timer{"Pipeline"};

        ++m_current_frame;
        reset_frame_stats();

        logger::info(R"(>>> Begin Frame: {})", m_current_frame);

        // Rebuild Scene
        {
            Timer timer{"Rebuild Scene"};
            // Trigger the rebuild_scene event, systems register their actions will be called here
            m_state = SimEngineState::RebuildScene;
            {
                event_rebuild_scene();

                // TODO: rebuild the vertex and surface info
                // m_global_vertex_manager->rebuild_vertex_info();
                // m_global_surface_manager->rebuild_surface_info();
            }

            // After the rebuild_scene event, the pending creation or deletion can be solved
            world().scene().solve_pending();

            // Update the diff parms
            update_diff_parm();
        }

        // Simulation:
        {
            Timer timer{"Simulation"};

            // 0. Process External Changes
            m_global_vertex_manager->update_attributes();
            [[maybe_unused]] AABB bbox =
                m_global_vertex_manager->compute_vertex_bounding_box();

            // 1. Record Friction Candidates at the beginning of the frame
            m_global_vertex_manager->record_prev_positions();
            record_friction_candidates();

            // 2. Predict Motion => x_tilde = x + v * dt
            m_state = SimEngineState::PredictMotion;
            // MUST step animation before predicting dof
            // some animation may provide information for DOF prediction
            step_animation_and_external_forces();
            m_time_integrator_manager->predict_dof();

            // 3. Adaptive Parameter Calculation
            detect_dcd_candidates();
            compute_adaptive_kappa();

            // 4. Nonlinear-Newton Iteration
            m_newton_tolerance_manager->pre_newton(m_current_frame);

            auto newton_max_iter = m_newton_max_iter->view()[0];
            auto newton_min_iter = m_newton_min_iter->view()[0];
            beta                 = 1.0;
            IndexT flag_rescue_iteration=-1;
            bool pilot_exact_active=false;
            IndexT pilot_exact_start=-1;
            const bool pilot_enabled=std::getenv("BUNNY_EVENT_FALLBACK")
                && std::string(std::getenv("BUNNY_EVENT_FALLBACK"))=="1";
            IndexT newton_iter   = 0;
            for(; newton_iter < newton_max_iter; ++newton_iter)
            {
                Timer timer{"Newton Iteration"};
                m_newton_iter             = newton_iter;
                m_frame_newton_iterations = newton_iter + 1;

                // 1) Compute animation substep ratio
                compute_animation_substep_ratio(newton_iter);


                // 2) Build Collision Pairs
                if(newton_iter > 0)
                    detect_dcd_candidates();


                // 3) Compute Dynamic Topo Effect Gradient and Hessian => G:Vector3, H:Matrix3x3
                //    - Contact Effect
                //    - Other DyTopo Effects
                m_state = SimEngineState::ComputeDyTopoEffect;
                compute_dytopo_effect();


                // 4) Solve Global Linear System => dx = A^-1 * b
                m_state = SimEngineState::SolveGlobalLinearSystem;
                {
                    Timer timer{"Solve Global Linear System"};
                    if(pilot_exact_active) _putenv_s("UIPC_PSQ_EXACT_ONCE","1");
                    logger::info("BUNNY_EVENT_MODE frame={} newton={} exact={}",m_current_frame,newton_iter,pilot_exact_active);
                    m_global_linear_system->solve();
                    m_frame_linear_solver_iterations +=
                        m_global_linear_system->last_solve_iterations();
                }


                // 5) Collect Vertex Displacements Globally
                m_global_vertex_manager->collect_vertex_displacements();

                // 7) Begin Line Search
                m_state = SimEngineState::LineSearch;
                {
                    Timer timer{"Line Search"};

                    // Reset Alpha
                    alpha = 1.0;

                    // Record Current State x to x_0
                    m_line_searcher->record_start_point();
                    m_global_vertex_manager->record_start_point();

                    if(m_dump_surface->view()[0])
                    {
                        dump_global_surface_pre_ccd(newton_iter);
                    }

                    // Stiff-GIPC design: cap the step with the feasible step
                    // over the active contact set first, then generate the
                    // swept-AABB trajectory candidates over the capped step
                    // (fewer candidates than over the full step)
                    alpha = feasible_step(alpha);

                    detect_trajectory_candidates(alpha);

                    // Compute Current Energy => E_0
                    Float E0 = m_line_searcher->compute_energy(true);  // initial energy

                    // CCD filter
                    alpha = filter_toi(alpha);

                    // CFL Condition (Stiff-GIPC design: the per-step speed cap
                    // applies only when some trajectory pair actually hits within
                    // this step; in free flight it must not bite)
                    if(ccd_alpha < 1.0)
                        alpha = cfl_condition(alpha);

                    const bool flag_adaptive_search=std::getenv("UIPC_PSQ_ADAPTIVE_SEARCH") && std::string(std::getenv("UIPC_PSQ_ADAPTIVE_SEARCH"))=="1";
                    const Float flag_search_cap=alpha;
                    Float flag_previous_alpha=0,flag_previous_delta=0;
                    // Line Search Iteration
                    bool  converged = convergence_check(newton_iter);
                    // Require at least three exact-gradient directions after switching.
                    if(pilot_exact_active && newton_iter-pilot_exact_start<2) converged=false;
                    SizeT line_search_iter = 0;
                    Float last_E           = E0;
                    for(; line_search_iter < m_line_searcher->max_iter(); ++line_search_iter)
                    {
                        Timer timer{"Line Search Iteration"};
                        m_line_search_iter = line_search_iter;
                        ++m_frame_line_search_trials;
                        m_frame_last_line_search_alpha = alpha;

                        // Compute Test Energy:
                        //  * Step Forward => x = x_0 + alpha * dx
                        //  * Compute New Energy => E
                        Float E = compute_energy(alpha);
                        last_E  = E;
                        if(flag_adaptive_search) logger::info("FLAG_SEARCH_TRIAL frame={} newton={} search={} alpha={:.17g} E0={:.17g} E={:.17g} cap={:.17g} converged={}",m_current_frame,newton_iter,line_search_iter,alpha,E0,E,flag_search_cap,converged);

                        #include "psq_converged_trial.inl"
                        // PSQ reject nonfinite trial before convergence acceptance
                        if(reject_inf_trials && std::isinf(E) && E>0)
                        {
                            logger::warn("PSQ_REJECT_INF_TRIAL frame={} newton={} search={} alpha={:.17g}",m_current_frame,newton_iter,line_search_iter,alpha);
                            alpha /= 2;
                            continue;
                        }

                        // To prevent numerical energy (fake-) increasing caused by tiny dx
                        

                        // Check Energy Decrease
                        // TODO: maybe better condition like Wolfe condition/Armijo condition in the future
                        bool energy_decrease = (E <= E0);

                        // Check Inversion
                        // TODO: Inversion check if needed
                        bool no_inversion = true;

                        bool success = energy_decrease && no_inversion;
                        if(success)
                            break;

                        // If not success, then shrink alpha
                        if(flag_adaptive_search) {
                            const Float next=flag_line_search::next(alpha,E-E0,flag_previous_alpha,flag_previous_delta);
                            logger::info("FLAG_SEARCH_SHRINK frame={} newton={} search={} alpha={:.17g} next={:.17g} delta={:.17g} previous_alpha={:.17g} previous_delta={:.17g}",m_current_frame,newton_iter,line_search_iter,alpha,next,E-E0,flag_previous_alpha,flag_previous_delta);
                            flag_previous_alpha=alpha;flag_previous_delta=E-E0;alpha=next;
                        } else alpha /= 2;
                    }

                    // Check Line Search Iteration: report warnings or throw exceptions if needed
                    if(reject_inf_trials && !std::isfinite(last_E))
                    {
                        compute_energy(0.0);
                        throw SimEngineException("PSQ trial energy stayed nonfinite after backtracking; restored current iterate");
                    }
                    if(newton_iter==flag_rescue_iteration) {
                        if(line_search_iter>=m_line_searcher->max_iter() || !std::isfinite(last_E) || last_E>E0) {
                            compute_energy(0.0);
                            throw SimEngineException("Flag exact-gradient rescue failed energy descent; current iterate restored");
                        }
                        logger::info("FLAG_PSQ_RESCUE_ACCEPT frame={} newton={} E0={:.17g} E1={:.17g} alpha={:.17g} ccd={:.17g}",m_current_frame,newton_iter,E0,last_E,m_frame_last_line_search_alpha,ccd_alpha);
                    }
// PSQ Algorithm 1: keep x_k when backtracking produces alpha < gamma.
const char* gamma_env = std::getenv("UIPC_PSQ_PREVIEW_MIN_ALPHA");
const double preview_gamma = gamma_env ? std::atof(gamma_env) : 0;
if(m_finite_element_method && animation_reach_target()
   && psq_preview::small_step(preview_requested,psq_preview_eligible(*m_finite_element_method),
                              converged,m_frame_last_line_search_alpha,preview_gamma))
{
    const Float retained=compute_energy(0.0);
    if(!std::isfinite(retained))throw SimEngineException("PSQ small-step preview has nonfinite retained energy");
    m_frame_preview_small_step = true;
    m_frame_converged = false;
    m_frame_hit_line_search_limit = line_search_iter >= m_line_searcher->max_iter();
    logger::info("PSQ_PREVIEW_SMALL_STEP frame={} newton={} rejected_alpha={:.17g} gamma={:.17g} retained_energy={:.17g}",
        m_current_frame,newton_iter,m_frame_last_line_search_alpha,preview_gamma,retained);
    m_frame_last_line_search_alpha=0;
    break;
}
// Rounded equality can accept a vanishing step before exhausting the budget.
// Opt-in probing also checks that case; it does NOT accept on small alpha alone.
// Three positive, resolved slopes must still establish non-descent.
const char* tiny_probe_env = std::getenv("UIPC_PSQ_PREVIEW_TINY_PROBE");
const bool tiny_probe = tiny_probe_env && std::string(tiny_probe_env)=="1"
                        && m_frame_last_line_search_alpha < 1e-8;
if(!pilot_exact_active && preview_requested && !converged
   && (line_search_iter >= m_line_searcher->max_iter() || tiny_probe)
   && animation_reach_target() && m_finite_element_method
   && psq_preview_eligible(*m_finite_element_method))
{
    const Float saved_alpha = m_frame_last_line_search_alpha;
    const Float swept_alpha = flag_adaptive_search ? flag_search_cap : std::ldexp(alpha, static_cast<int>(line_search_iter));
    const Float base = compute_energy(0.0);
    double h[3]={swept_alpha*1e-3,swept_alpha*1e-4,swept_alpha*1e-5};
    double e[3]={compute_energy(h[0]),compute_energy(h[1]),compute_energy(h[2])};
    auto decision=psq_preview::assess(base,last_E,h,e);
    // Refine only resolved positive but scale-dependent slopes. Keep the same
    // noise floor and 25% consistency test; diagnostic probes are never accepted updates.
    if(std::getenv("UIPC_PSQ_REFINE_PROBES") && std::string(std::getenv("UIPC_PSQ_REFINE_PROBES"))=="1") {
        for(int refinement=1; refinement<=3 && decision.resolved && !decision.consistent; ++refinement) {
            logger::info("FLAG_PROBE_REFINE frame={} newton={} refinement={} old_h1={:.17g} old_slope1={:.17g} old_slope2={:.17g} old_slope3={:.17g}",m_current_frame,newton_iter,refinement,h[0],decision.slopes[0],decision.slopes[1],decision.slopes[2]);
            h[0]=h[1];h[1]=h[2];h[2]*=.1;
            e[0]=e[1];e[1]=e[2];e[2]=compute_energy(h[2]);
            decision=psq_preview::assess(base,last_E,h,e);
            logger::info("FLAG_PROBE_REFINED frame={} newton={} refinement={} resolved={} consistent={} h1={:.17g} h2={:.17g} h3={:.17g} E0={:.17g} e1={:.17g} e2={:.17g} e3={:.17g}",m_current_frame,newton_iter,refinement,decision.resolved,decision.consistent,h[0],h[1],h[2],base,e[0],e[1],e[2]);
        }
    }
    if(decision.consistent)
    {
        // Restore x_k, NOT the non-descending trial or the time-step start.
        const Float retained=compute_energy(0.0);
        if(!std::isfinite(retained) || std::abs(retained-base)>decision.noise)
            throw SimEngineException("PSQ preview failed to restore current iterate");
        // A single exact-gradient retry only when the time step made no update.
        // All ordinary feasibility, CCD, line-search and convergence checks remain.
        if(newton_iter==0 && std::getenv("UIPC_PSQ_RESCUE")
           && std::string(std::getenv("UIPC_PSQ_RESCUE"))=="1") {
            flag_rescue_iteration=newton_iter+1;
            _putenv_s("UIPC_PSQ_EXACT_ONCE","1");
            logger::info("FLAG_PSQ_RESCUE frame={} newton=0 reason=verified_non_descent_at_step_start",m_current_frame);
            continue;
        }
        if(pilot_enabled && !pilot_exact_active) {
            const Float pilot_retained=compute_energy(0.0);
            const Float pilot_noise=64*std::numeric_limits<Float>::epsilon()*std::max(std::abs(E0),Float(1e-30));
            if(!std::isfinite(pilot_retained) || std::abs(pilot_retained-E0)>pilot_noise)
                throw SimEngineException("Event fallback failed to restore current iterate");
            pilot_exact_active=true; pilot_exact_start=newton_iter+1;
            m_frame_hit_line_search_limit=false;
            m_frame_last_line_search_alpha=0;
            logger::info("BUNNY_EVENT_TRIGGER frame={} newton={} reason=finite_difference E0={:.17g} retained={:.17g}",
                         m_current_frame,newton_iter,E0,pilot_retained);
            continue;
        }
        m_frame_preview_non_descent = true;
        m_frame_converged = false;
        m_frame_hit_line_search_limit = line_search_iter >= m_line_searcher->max_iter();
        m_frame_last_line_search_alpha = 0;
        logger::info("PSQ_PREVIEW_STOP frame={} reason=preview_non_descent newton={} alpha=0 E_retained={:.17g} E_base={:.17g} E_last={:.17g} h1={:.17g} h2={:.17g} h3={:.17g} e1={:.17g} e2={:.17g} e3={:.17g} slope1={:.17g} slope2={:.17g} slope3={:.17g} energy_noise={:.17g}",
            m_current_frame,newton_iter,retained,base,last_E,h[0],h[1],h[2],e[0],e[1],e[2],decision.slopes[0],decision.slopes[1],decision.slopes[2],decision.noise);
        break; // exits Newton loop; ordinary velocity update follows
    }
    else
    {
        if(std::getenv("UIPC_PSQ_UNRESOLVED_RESCUE")
           && std::string(std::getenv("UIPC_PSQ_UNRESOLVED_RESCUE"))=="1") {
            const Float retained=compute_energy(0.0);
            if(!std::isfinite(retained) || std::abs(retained-base)>decision.noise)
                throw SimEngineException("Flag unresolved rescue failed to restore current iterate");
            flag_rescue_iteration=newton_iter+1;
            _putenv_s("UIPC_PSQ_EXACT_ONCE","1");
            logger::info("FLAG_PSQ_UNRESOLVED_RESCUE frame={} newton={} retry={} E_base={:.17g} E_retained={:.17g} noise={:.17g} resolved={} consistent={}",m_current_frame,newton_iter,flag_rescue_iteration,base,retained,decision.noise,decision.resolved,decision.consistent);
            continue; // new exact gradient at restored x_k; ordinary feasibility and 8 trials
        }
        compute_energy(saved_alpha);
        logger::info("PSQ_PREVIEW_REJECT frame={} newton={} resolved={} consistent={} slope1={:.17g} slope2={:.17g} slope3={:.17g}",
            m_current_frame,newton_iter,decision.resolved,decision.consistent,decision.slopes[0],decision.slopes[1],decision.slopes[2]);
    }
}

#include <finite_element/constitutions/psq_direction_audit.inl>
#include <finite_element/constitutions/psq_consistency_audit.inl>
#include <finite_element/constitutions/psq_force_update.inl>
                    // PD 2024, section 4.6: a preview may stop if the back-solved direction
// no longer provides descent. Differentiate the *objective* for this test.
// The Newton direction itself was computed using the frozen offset.
if(!pilot_exact_active && line_search_iter >= m_line_searcher->max_iter() && !converged && preview_requested
   && m_finite_element_method && psq_preview_eligible(*m_finite_element_method)
   && animation_reach_target() && std::getenv("UIPC_PSQ_FROZEN_ONLY"))
{
    const Float retained = compute_energy(0.0);
    const Float noise = 64 * std::numeric_limits<Float>::epsilon() * std::max(std::abs(E0), Float(1e-30));
    if(!std::isfinite(retained) || std::abs(retained - E0) > noise)
    {
        status().push_back(core::EngineStatus::error("Bunny derivative test could not restore its base point"));
        return;
    }

    auto& system = m_global_linear_system->m_impl;
    vector<Float> solution(system.x.size()), gradient(system.x.size());
    system.x.cview().buffer_view().copy_to(solution.data());
    GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo effect;
    effect.component_flags(EnergyComponentFlags::All);
    effect.gradient_only(true);
    m_global_dytopo_effect_manager->compute_dytopo_effect(effect);
    cuda_tool::DeviceDenseVector<Float> exact;
    exact.resize(system.x.size());
    exact.view().buffer_view().fill(0);
    GlobalLinearSystem::ComputeGradientInfo info;
    info.buffer_view(exact.view());
    info.flags(EnergyComponentFlags::All);
    _putenv_s("UIPC_PSQ_EXACT_ONCE", "1");
    m_global_linear_system->compute_gradient(info);
    _putenv_s("UIPC_PSQ_EXACT_ONCE", "0");
    exact.cview().buffer_view().copy_to(gradient.data());

    Float slope = 0, correction = 0, magnitude = 0;
    for(SizeT i = 0; i < gradient.size(); ++i)
    {
        // FEM applies minus the back-solved vector as its displacement.
        const Float term = -gradient[i] * solution[i];
        const Float adjusted = term - correction;
        const Float sum = slope + adjusted;
        correction = (sum - slope) - adjusted;
        slope = sum;
        magnitude += std::abs(term);
    }
    const Float bound = 128 * std::numeric_limits<Float>::epsilon() * magnitude;
    logger::info("BUNNY_PREVIEW_DERIVATIVE frame={} newton={} slope={:.17g} magnitude={:.17g} bound={:.17g} E0={:.17g} retained={:.17g}",
                 m_current_frame, newton_iter, slope, magnitude, bound, E0, retained);
    if(std::isfinite(slope) && std::isfinite(magnitude) && slope > bound)
    {
        if(pilot_enabled && !pilot_exact_active) {
            const Float pilot_retained=compute_energy(0.0);
            const Float pilot_noise=64*std::numeric_limits<Float>::epsilon()*std::max(std::abs(E0),Float(1e-30));
            if(!std::isfinite(pilot_retained) || std::abs(pilot_retained-E0)>pilot_noise)
                throw SimEngineException("Event fallback failed to restore current iterate");
            pilot_exact_active=true; pilot_exact_start=newton_iter+1;
            m_frame_hit_line_search_limit=false;
            m_frame_last_line_search_alpha=0;
            logger::info("BUNNY_EVENT_TRIGGER frame={} newton={} reason=analytic E0={:.17g} retained={:.17g}",
                         m_current_frame,newton_iter,E0,pilot_retained);
            continue;
        }
        m_frame_preview_non_descent = true;
        m_frame_converged = false;
        m_frame_hit_line_search_limit = true;
        m_frame_last_line_search_alpha = 0;
        break;
    }
}

                    // Continue the coarse frozen-direction backtracking sequence until it either
// lowers the objective or rounds back to the starting positions for every DOF.
if(!pilot_exact_active && line_search_iter >= m_line_searcher->max_iter() && !converged && preview_requested
   && m_finite_element_method && psq_preview_eligible(*m_finite_element_method)
   && animation_reach_target() && std::getenv("UIPC_PSQ_FROZEN_ONLY"))
{
    const auto start_positions = m_finite_element_method->x_temps();
    vector<Vector3> base(start_positions.size()), trial(base.size());
    start_positions.copy_to(base.data());
    bool moved_downhill = false, exhausted = false;
    Float trial_alpha = alpha;
    for(int retry = 0; trial_alpha > 0 && trial_alpha < m_frame_last_line_search_alpha;
        ++retry, trial_alpha *= 0.5)
    {
        const Float energy = compute_energy(trial_alpha);
        ++m_frame_line_search_trials;
        m_frame_last_line_search_alpha = trial_alpha;
        const bool decrease = std::isfinite(energy) && energy < E0;
        if(decrease || retry >= 24)
        {
            m_finite_element_method->xs().copy_to(trial.data());
            exhausted = std::equal(base.begin(), base.end(), trial.begin(),
                                   [](const Vector3& a, const Vector3& b) {
                                       return (a.array() == b.array()).all();
                                   });
        }
        moved_downhill = decrease && !exhausted;
        logger::info("BUNNY_FROZEN_SEARCH frame={} newton={} retry={} alpha={:.17g} cap={:.17g} E0={:.17g} E={:.17g} accepted={} identical={}",
                     m_current_frame, newton_iter, retry, trial_alpha, flag_search_cap, E0, energy, moved_downhill, exhausted);
        if(moved_downhill || exhausted)
            break;
    }
    if(moved_downhill)
        continue;
    const Float retained = compute_energy(0);
    const Float noise = 64 * std::numeric_limits<Float>::epsilon() * std::max(std::abs(E0), Float(1e-30));
    const bool restored = std::isfinite(retained) && std::abs(retained - E0) <= noise;
    m_frame_last_line_search_alpha = 0;
    m_frame_hit_line_search_limit = true;
    if(restored && exhausted)
    {
        if(pilot_enabled && !pilot_exact_active) {
            const Float pilot_retained=compute_energy(0.0);
            const Float pilot_noise=64*std::numeric_limits<Float>::epsilon()*std::max(std::abs(E0),Float(1e-30));
            if(!std::isfinite(pilot_retained) || std::abs(pilot_retained-E0)>pilot_noise)
                throw SimEngineException("Event fallback failed to restore current iterate");
            pilot_exact_active=true; pilot_exact_start=newton_iter+1;
            m_frame_hit_line_search_limit=false;
            m_frame_last_line_search_alpha=0;
            logger::info("BUNNY_EVENT_TRIGGER frame={} newton={} reason=representable E0={:.17g} retained={:.17g}",
                         m_current_frame,newton_iter,E0,pilot_retained);
            continue;
        }
        m_frame_preview_non_descent = true;
        m_frame_converged = false;
        logger::info("BUNNY_PREVIEW_STAGNATION frame={} newton={} vertices={} alpha={:.17g} E0={:.17g} retained={:.17g}",
                     m_current_frame, newton_iter, base.size(), trial_alpha, E0, retained);
        break;
    }
    status().push_back(core::EngineStatus::error("FEM backtracking failed to certify a retained preview state"));
    return;
}

                    #include "bunny_exhausted_search.inl"
                    if(!forced_iteration) check_line_search_iter(line_search_iter, E0, last_E);
                    if(forced_iteration) break;

                    // newton/min_iter is a pure hard floor (default 0 = off);
                    // the semi-implicit Kmin lives in newton/semi_implicit/K_min
                    bool terminated = converged && (newton_iter >= newton_min_iter);
                    if(terminated)
                    {
                        m_frame_converged = true;
                        break;
                    }
                }
            }

            // 5. Update Velocity => v = (x - x_0) / dt
            m_state = SimEngineState::UpdateVelocity;
            {
                Timer timer{"Update Velocity"};
                m_time_integrator_manager->update_state();
            }

            // Check Newton Iteration
            // report warnings or throw exceptions if needed
            check_newton_iter(newton_iter);
            m_frame_completed = true;
        }

        logger::info("<<< End Frame: {}", m_current_frame);
    };

    try
    {
        pipeline();
    }
    catch(const SimEngineException& e)
    {
        logger::error("Engine Advance Error: {}", e.what());
        status().push_back(core::EngineStatus::error(e.what()));
    }
    catch(const std::exception& e)
    {
        UIPC_ASSERT(false, "Unexpected Exception: {}", e.what());
    }
}
}  // namespace uipc::backend::cuda
