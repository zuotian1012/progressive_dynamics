// Explicit user experiment: accept a feasible update after the backtracking
// budget, even when it does not lower energy. This is not Newton convergence.
bool forced_iteration = false;
if(force_updates && line_search_iter >= m_line_searcher->max_iter())
{
    Float feasible_alpha = std::ldexp(alpha, static_cast<int>(line_search_iter));
    Float forced_energy = compute_energy(feasible_alpha);
    if(!(feasible_alpha > 0) || !std::isfinite(forced_energy))
        throw SimEngineException("Forced update rejected: nonfinite or infeasible trial");
    m_frame_hit_line_search_limit = true;
    m_frame_last_line_search_alpha = feasible_alpha;
    m_frame_converged = false;
    forced_frame = true;
    forced_iteration = true;
    logger::warn("PSQ_FORCE_ACCEPT frame={} reason=line_search_limit newton={} alpha={:.17g} E0={:.17g} E_applied={:.17g}",
                 m_current_frame, newton_iter, feasible_alpha, E0, forced_energy);
}
