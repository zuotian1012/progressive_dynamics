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
if(preview_requested && !converged
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
