// The standard search and any permitted preview stop have already run.
// A failed preview test does not authorize moving to its last trial.
if(!forced_iteration && line_search_iter >= m_line_searcher->max_iter())
{
    Float trial_alpha = alpha;  // next backtracking proposal, inside the swept CCD bound
    const Float noise = 64 * std::numeric_limits<Float>::epsilon()
                        * std::max(std::abs(E0), Float(1e-30));
    bool accepted = false;
    for(int retry = 0; retry < 24 && trial_alpha > 0; ++retry, trial_alpha *= 0.5)
    {
        if(!(trial_alpha < m_frame_last_line_search_alpha && trial_alpha <= flag_search_cap))
            break;
        const Float energy = compute_energy(trial_alpha);
        ++m_frame_line_search_trials;
        m_frame_last_line_search_alpha = trial_alpha;
        // Use representable objective decrease. The restoration tolerance
        // measures repeat evaluation at x_k, not a required decrease at a trial.
        accepted = std::isfinite(energy) && energy < E0;
        logger::info("BUNNY_SEARCH_RETRY frame={} newton={} retry={} alpha={:.17g} cap={:.17g} E0={:.17g} E={:.17g} noise={:.17g} accepted={}",
                     m_current_frame, newton_iter, retry, trial_alpha, flag_search_cap, E0, energy, noise, accepted);
        if(accepted)
            break;
    }
    if(accepted)
        continue;  // begin a fresh Newton iteration at the verified trial

    const Float retained = compute_energy(0);
    m_frame_last_line_search_alpha = 0;
    m_frame_hit_line_search_limit = true;
    const bool restored = std::isfinite(retained) && std::abs(retained - E0) <= noise;
    auto message = fmt::format("BUNNY_SEARCH_FAILED frame={} newton={} E0={:.17g} retained={:.17g} restored={}",
                               m_current_frame, newton_iter, E0, retained, restored);
    logger::error("{}", message);
    status().push_back(core::EngineStatus::error(message));
    return;
}
