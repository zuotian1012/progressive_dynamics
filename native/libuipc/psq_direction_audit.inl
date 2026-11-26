// Optional failure-only diagnostic: no trial is accepted by this block.
if(line_search_iter >= m_line_searcher->max_iter()
   && std::getenv("UIPC_PSQ_DIRECTION_AUDIT"))
{
    auto& system = m_global_linear_system->m_impl;
    vector<Float> g(system.b.size()), solution(system.x.size());
    CUDA_TOOL_CHECK(cudaMemcpy(g.data(), system.b.cview().data(),
                               g.size()*sizeof(Float), cudaMemcpyDeviceToHost));
    CUDA_TOOL_CHECK(cudaMemcpy(solution.data(), system.x.cview().data(),
                               solution.size()*sizeof(Float), cudaMemcpyDeviceToHost));
    Float predicted=0;
    for(SizeT i=0;i<g.size();++i) predicted-=g[i]*solution[i];
    // Positive substeps only: stay within the previously checked CCD sweep.
    Float base=compute_energy(0.0);
    Float swept_alpha=std::ldexp(alpha,static_cast<int>(line_search_iter));
    Float strain_alpha=m_finite_element_method ? psq_feasible_step(*m_finite_element_method,1.0):1.0;
    logger::info("PSQ_AUDIT_CAPS frame={} newton={} strain_alpha={:.17g} swept_alpha={:.17g} ccd_alpha={:.17g} cfl_alpha={:.17g}",
        m_current_frame,newton_iter,strain_alpha,swept_alpha,ccd_alpha,cfl_alpha);
    for(Float fraction : {1e-2,1e-3,1e-4,1e-5})
    {
        Float probe=fraction*swept_alpha;
        Float trial=compute_energy(probe);
        logger::info("PSQ_DIRECTION_AUDIT frame={} newton={} alpha={} predicted_slope={:.17g} actual_slope={:.17g} base={:.17g} trial={:.17g}",
            m_current_frame,newton_iter,probe,predicted,(trial-base)/probe,base,trial);
    }
    compute_energy(0.0);
    if(m_finite_element_method)
    {
        psq_diagnostic_freeze_offset(*m_finite_element_method,true);
        Float frozen_base=compute_energy(0.0);
        for(Float fraction : {1e-2,1e-3,1e-4,1e-5})
        {
            Float probe=fraction*swept_alpha;
            Float trial=compute_energy(probe);
            logger::info("PSQ_FROZEN_AUDIT frame={} newton={} alpha={} predicted_slope={:.17g} actual_slope={:.17g} base={:.17g} trial={:.17g}",
                m_current_frame,newton_iter,probe,predicted,(trial-frozen_base)/probe,frozen_base,trial);
        }
        psq_diagnostic_freeze_offset(*m_finite_element_method,false);
        compute_energy(0.0);
    }
}
