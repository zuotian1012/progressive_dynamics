// Snapshot successive assembled gradients; differences isolate drive reporters
// and dynamic topology (contact) contributions without changing the solve.
auto& audit_engine=affine_body_dynamics->engine();
SizeT audit_frame=audit_engine.frame(),audit_iter=audit_engine.newton_iter();
bool audit_abd=std::getenv("UIPC_PSQ_CONSISTENCY_AUDIT") && audit_frame>=400 && audit_frame<=500;
auto audit_gradient=[&](const char* stage) {
    if(!audit_abd)return;
    vector<Float> values(abd().body_count()*12);
    CUDA_TOOL_CHECK(cudaMemcpy(values.data(),info.gradients().buffer_view().data(),values.size()*sizeof(Float),cudaMemcpyDeviceToHost));
    for(SizeT body=0;body<abd().body_count();++body) {
        const Float* v=values.data()+12*body;
        logger::info("PSQ_ABD_COMPONENT frame={} newton={} body={} stage={} g=[{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}]",audit_frame,audit_iter,body,stage,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11]);
    }
};
