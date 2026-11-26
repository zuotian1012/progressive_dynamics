// Continue the coarse frozen-direction backtracking sequence until it either
// lowers the objective or rounds back to the starting positions for every DOF.
if(line_search_iter >= m_line_searcher->max_iter() && !converged && preview_requested
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
        m_frame_preview_non_descent = true;
        m_frame_converged = false;
        logger::info("BUNNY_PREVIEW_STAGNATION frame={} newton={} vertices={} alpha={:.17g} E0={:.17g} retained={:.17g}",
                     m_current_frame, newton_iter, base.size(), trial_alpha, E0, retained);
        break;
    }
    status().push_back(core::EngineStatus::error("FEM backtracking failed to certify a retained preview state"));
    return;
}
