// A small Newton decrement certifies x_k. A trial x_k + alpha*dx can
// still increase the energy when the PSQ Hessian is only approximate.
// Backtrack that trial normally. On exhaustion keep the certified x_k.
if(converged && newton_iter != flag_rescue_iteration)
{
    if(!std::isfinite(E0))
        throw SimEngineException("Nonfinite energy at a converged Newton iterate");
    if(std::isfinite(E) && E <= E0)
        break;
    if(line_search_iter + 1 == m_line_searcher->max_iter())
    {
        const Float retained = compute_energy(0.0);
        const Float noise = 64 * std::numeric_limits<Float>::epsilon()
                            * std::max(std::abs(E0), Float(1e-30));
        if(!std::isfinite(retained) || std::abs(retained-E0) > noise)
            throw SimEngineException("Failed to restore converged Newton iterate");
        last_E = retained;
        m_frame_last_line_search_alpha = 0;
        logger::info("PSQ_CONVERGED_RESTORE frame={} newton={} E0={:.17g} trial={:.17g} retained={:.17g} noise={:.17g}",
                     m_current_frame, newton_iter, E0, E, retained, noise);
        break;
    }
}
