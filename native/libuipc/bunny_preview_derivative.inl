// PD 2024, section 4.6: a preview may stop if the back-solved direction
// no longer provides descent. Differentiate the *objective* for this test.
// The Newton direction itself was computed using the frozen offset.
if(line_search_iter >= m_line_searcher->max_iter() && !converged && preview_requested
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
        m_frame_preview_non_descent = true;
        m_frame_converged = false;
        m_frame_hit_line_search_limit = true;
        m_frame_last_line_search_alpha = 0;
        break;
    }
}
