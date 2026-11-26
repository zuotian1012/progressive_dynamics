// Observational post-search probe. Restore the exact last tested alpha, which
// differs from alpha after an exhausted backtracking loop (alpha was halved).
if(std::getenv("UIPC_PSQ_CONSISTENCY_AUDIT"))
{
    bool target=m_current_frame==26 || m_current_frame==426 || m_current_frame==460 || m_current_frame==490;
    bool sample=newton_iter==0 || newton_iter==1 || newton_iter==5 || newton_iter==10 || newton_iter==20 || newton_iter==50 || newton_iter%100==0;
    bool failure=line_search_iter>=m_line_searcher->max_iter();
    if((target && sample) || (failure && newton_iter%100==0) || newton_iter==1023)
    {
        auto& system=m_global_linear_system->m_impl;
        vector<Float> g(system.b.size()),solution(system.x.size());
        CUDA_TOOL_CHECK(cudaMemcpy(g.data(),system.b.cview().data(),g.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        CUDA_TOOL_CHECK(cudaMemcpy(solution.data(),system.x.cview().data(),solution.size()*sizeof(Float),cudaMemcpyDeviceToHost));
        Float predicted=0;for(SizeT i=0;i<g.size();++i)predicted-=g[i]*solution[i];
        Float restore_alpha=m_frame_last_line_search_alpha;
        Float swept_alpha=std::ldexp(alpha,static_cast<int>(line_search_iter));
        Float base=compute_energy(0.0);
        for(Float fraction : {1e-2,1e-3,1e-4,1e-5,1e-6}) {
            Float h=fraction*swept_alpha,trial=compute_energy(h);
            logger::info("PSQ_CONSISTENCY frame={} newton={} frozen=0 h={:.17g} predicted={:.17g} measured={:.17g} base={:.17g} trial={:.17g}",m_current_frame,newton_iter,h,predicted,(trial-base)/h,base,trial);
        }
        compute_energy(0.0);
        if(m_finite_element_method) {
            psq_diagnostic_freeze_offset(*m_finite_element_method,true);
            Float frozen_base=compute_energy(0.0);
            for(Float fraction : {1e-2,1e-3,1e-4,1e-5,1e-6}) {
                Float h=fraction*swept_alpha,trial=compute_energy(h);
                logger::info("PSQ_CONSISTENCY frame={} newton={} frozen=1 h={:.17g} predicted={:.17g} measured={:.17g} base={:.17g} trial={:.17g}",m_current_frame,newton_iter,h,predicted,(trial-frozen_base)/h,frozen_base,trial);
            }
            psq_diagnostic_freeze_offset(*m_finite_element_method,false);
        }
        Float restored=compute_energy(restore_alpha);
        logger::info("PSQ_CONSISTENCY_RESTORE frame={} newton={} alpha={:.17g} before={:.17g} after={:.17g} error={:.17g}",m_current_frame,newton_iter,restore_alpha,last_E,restored,restored-last_E);
        if(!std::isfinite(restored) || std::abs(restored-last_E)>1e-10*std::max(Float(1),std::abs(last_E)))
            throw SimEngineException("PSQ consistency audit failed to restore trial energy");
    }
}
