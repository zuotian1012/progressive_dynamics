"""Install the bounded PSQ adapter into the isolated source checkout only."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/'external/libuipc-psq-src'
BACK = SRC/'src/backends/cuda'
HERE = Path(__file__).resolve().parent

def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError(f'Unexpected upstream source at {before[:80]!r}: {text.count(before)}')
    return text.replace(before, after)

def preserve_contact_on_velocity_write(source):
    notification = 'm_vertex_reporter.request_attribute_update();'
    write_position = '        x_subview.copy_from(pos_view.data());'
    updated = (write_position + '\n'
               '        // A position edit invalidates lagged contact geometry. A velocity-only\n'
               '        // write changes the BDF predictor and must retain the friction set.\n'
               '        ' + notification)
    if source.count(notification) != 1:
        raise ValueError('Unexpected state accessor notification count')
    if updated in source:
        return source
    unconditional = '    // request the vertex reporter to update attributes\n    ' + notification + '\n'
    return replace(replace(source, unconditional, ''), write_position, updated)

def main():
    accessor = BACK/'finite_element/finite_element_state_accessor_feature.cu'
    source = accessor.read_text()
    corrected = preserve_contact_on_velocity_write(source)
    if corrected != source:
        accessor.write_text(corrected)
    for name in ['psq_proxy.h','psq_proxy.cu','psq_kernels.cuh','psq_strain.cuh','psq_step_cap.h','psq_direction_audit.inl','psq_consistency_audit.inl','psq_abd_audit.inl','psq_force_update.inl','psq_preview_stop.inl','psq_preview_decision.h','psq_trial_energy.h']:
        shutil.copy2(HERE/name, BACK/'finite_element/constitutions'/name)
    path=BACK/'finite_element/constitutions/neo_hookean_shell_2d.cu'
    s=path.read_text()
    if '// PSQ adapter' not in s:
        s='#include "psq_proxy.h" // PSQ adapter\n'+s
        s=replace(s,'Float                             dt,\n        int',
            'Float                             dt,\n        bool barrier, bool replace_elastic, psq_native::StrainParameters params,\n        int')
        s=replace(s,'NH::E(E, lambda, mu, X, IB);','NH::E(E, lambda, mu, X, IB);\n        if(replace_elastic) E=0;\n        if(barrier) E+=psq_native::strain(X,IB,params);')
        s=replace(s,'bool                                   gradient_only,\n        int',
            'bool                                   gradient_only,\n        bool barrier, bool replace_elastic, psq_native::StrainParameters params,\n        int')
        s=replace(s,'G *= Vdt2;', '''if(replace_elastic) G.setZero();
        Vector9 strain_G=Vector9::Zero(); Matrix9x9 strain_H=Matrix9x9::Zero();
        if(barrier) psq_native::strain(X,IB,params,&strain_G,gradient_only?nullptr:&strain_H);
        G += strain_G;
        G *= Vdt2;''')
        s=replace(s,'make_spd(H);','H += strain_H;\n        make_spd(H);')
        s=replace(s,'SimSystemSlot<FiniteElementMethod> fem;','SimSystemSlot<FiniteElementMethod> fem;\n    PsqProxy proxy;')
        s=replace(s,'prims, x_bars, inv_B_matrices.view(), n);\n        }',
                    'prims, x_bars, inv_B_matrices.view(), n);\n        }\n        proxy.init(*fem.view(),prims,inv_B_matrices.cview(),h_lambdas,h_mus);')
        s=replace(s,'info.energy_count(mus.size());','info.energy_count(mus.size()+proxy.energy_count());')
        s=replace(s,'info.gradient_count(mus.size() * StencilSize);','info.gradient_count(mus.size() * StencilSize+proxy.gradient_count());')
        s=replace(s,'info.dt(),\n                n);','info.dt(),\n                proxy.active(),proxy.replaces_elastic(),proxy.strain_parameters(),\n                n);\n            proxy.energy(info.xs(),info.energies().subview(n,proxy.energy_count()),info.dt());')
        s=replace(s,'info.gradient_only(),\n                n);','info.gradient_only(),\n                proxy.active(),proxy.replaces_elastic(),proxy.strain_parameters(),\n                n);\n            proxy.gradient(info.xs(),info.gradients().subview(n*StencilSize,proxy.gradient_count()),info.dt());')
        path.write_text(s)
    path=BACK/'finite_element/constitutions/discrete_shell_bending.cu'
    s=path.read_text()
    if '// PSQ adapter' not in s:
        s='#include "psq_proxy.h" // PSQ adapter\n'+s
        s=replace(s,'Float                           dt,\n        int','Float                           dt,\n        bool replace_elastic,\n        int')
        s=replace(s,'energies(I) = E * V_bar * dt * dt;','energies(I) = replace_elastic ? 0.0 : E * V_bar * dt * dt;')
        s=replace(s,'bool                                   gradient_only,\n        int','bool                                   gradient_only,\n        bool replace_elastic,\n        int')
        s=replace(s,'G12 *= Vdt2;','if(replace_elastic) G12.setZero();\n        G12 *= Vdt2;')
        s=replace(s,'info.dt(),\n                n);','info.dt(),\n                psq_replaces_bending(),\n                n);')
        s=replace(s,'info.gradient_only(),\n                n);','info.gradient_only(),\n                psq_replaces_bending(),\n                n);')
        path.write_text(s)
    path=BACK/'engine/advance_ipc.cu';s=path.read_text()
    if '// PSQ adapter' not in s:
        s='#include <finite_element/finite_element_method.h> // PSQ adapter\n#include <finite_element/constitutions/psq_proxy.h>\n'+s
        s=replace(s,'auto feasible_step = [this](Float alpha)\n    {','auto feasible_step = [this](Float alpha)\n    {\n        if(m_finite_element_method) alpha=psq_feasible_step(*m_finite_element_method,alpha);')
        path.write_text(s)
    s=path.read_text()
    s=s.replace('psq_feasible_step(*m_finite_element_method,alpha);','psq_feasible_step(*m_finite_element_method,alpha,m_current_frame,m_newton_iter);')
    path.write_text(s)
    if 'psq_direction_audit.inl' not in s:
        s='#include <cstdlib>\n#include <cmath>\n'+s
        s=replace(s,'                    check_line_search_iter(line_search_iter, E0, last_E);',
            '#include <finite_element/constitutions/psq_direction_audit.inl>\n                    check_line_search_iter(line_search_iter, E0, last_E);')
        path.write_text(s)
    s=path.read_text()
    if 'psq_force_update.inl' not in s:
        s=replace(s,'    Float cfl_alpha = 1.0;', '''    Float cfl_alpha = 1.0;
    const char* force_env = std::getenv("UIPC_PSQ_FORCE_UPDATE");
    const bool force_updates = force_env && std::string(force_env)=="1" && psq_replaces_bending();
    bool forced_frame = false;''')
        s=replace(s,'auto check_newton_iter = [this](IndexT newton_iter_after_loop)\n    {',
            'auto check_newton_iter = [this, &forced_frame, force_updates](IndexT newton_iter_after_loop)\n    {\n        if(forced_frame) return;')
        s=replace(s,'            m_frame_hit_newton_limit = true;', '''            m_frame_hit_newton_limit = true;
            if(force_updates)
            {
                forced_frame = true;
                m_frame_converged = false;
                logger::warn("PSQ_FORCE_ACCEPT frame={} reason=newton_limit newton={} alpha={:.17g}",
                             m_current_frame, newton_iter_after_loop-1, m_frame_last_line_search_alpha);
                return;
            }''')
        s=replace(s,'                    check_line_search_iter(line_search_iter, E0, last_E);',
            '#include <finite_element/constitutions/psq_force_update.inl>\n                    if(!forced_iteration) check_line_search_iter(line_search_iter, E0, last_E);\n                    if(forced_iteration) break;')
        path.write_text(s)
    # Explicit coarse-preview status, separate from strict convergence/force acceptance.
    path=BACK/'engine/advance_ipc.cu';s=path.read_text()
    if 'psq_preview_decision.h' not in s:
        s='#include <finite_element/constitutions/psq_preview_decision.h>\n'+s
        path.write_text(s)
    if 'psq_preview_stop.inl' not in s:
        s='#include <limits>\n'+s
        s=replace(s,'    bool forced_frame = false;', '''    bool forced_frame = false;
    const char* preview_env = std::getenv("UIPC_PSQ_PREVIEW_STOP");
    const bool preview_requested = preview_env && std::string(preview_env)=="1";
    if(preview_requested && force_updates)
        throw SimEngineException("PSQ preview and forced updates are mutually exclusive");''')
        s=replace(s,'        if(forced_frame) return;','        if(forced_frame || m_frame_preview_non_descent) return;')
        s=replace(s,'#include <finite_element/constitutions/psq_direction_audit.inl>',
            '#include <finite_element/constitutions/psq_preview_stop.inl>\n#include <finite_element/constitutions/psq_direction_audit.inl>')
        path.write_text(s)
    path=BACK/'sim_engine.h';s=path.read_text()
    if 'm_frame_preview_non_descent' not in s:
        s=replace(s,'    bool  m_frame_converged                = false;',
            '    bool  m_frame_converged                = false;\n    bool  m_frame_preview_non_descent      = false;')
        path.write_text(s)
    path=BACK/'engine/sim_engine.cu';s=path.read_text()
    if 'm_frame_preview_non_descent' not in s:
        s=replace(s,'    m_frame_converged                = false;',
            '    m_frame_converged                = false;\n    m_frame_preview_non_descent      = false;')
        s=replace(s,'    stats["converged"]                = m_frame_converged;',
            '    stats["converged"]                = m_frame_converged;\n    stats["preview_non_descent"] = m_frame_preview_non_descent;\n    stats["termination_reason"] = m_frame_preview_non_descent ? "preview_non_descent" : (m_frame_converged ? "newton_converged" : "not_converged");')
        path.write_text(s)
    path=BACK/'engine/advance_ipc.cu';s=path.read_text()
    if 'psq_consistency_audit.inl' not in s:
        s=replace(s,'#include <finite_element/constitutions/psq_direction_audit.inl>',
            '#include <finite_element/constitutions/psq_direction_audit.inl>\n#include <finite_element/constitutions/psq_consistency_audit.inl>')
        path.write_text(s)
    path=BACK/'affine_body/abd_linear_subsystem.cu';s=path.read_text()
    if 'psq_abd_audit.inl' not in s:
        s='#include <cstdlib>\n#include <uipc/common/log.h>\n'+s
        s=replace(s,'    // 1) Static Topo Effect: Kinetic + Shape + Other Reporters',
            '#include <finite_element/constitutions/psq_abd_audit.inl>\n    // 1) Static Topo Effect: Kinetic + Shape + Other Reporters')
        s=replace(s,'        _assemble_kinetic_shape(hess_offset, info);',
            '        _assemble_kinetic_shape(hess_offset, info);\n        audit_gradient("kinetic_shape");')
        s=replace(s,'        _assemble_reporters(hess_offset, info);',
            '        _assemble_reporters(hess_offset, info);\n        audit_gradient("plus_drive");')
        s=replace(s,'    _assemble_dytopo_effect(hess_offset, info);',
            '    _assemble_dytopo_effect(hess_offset, info);\n    audit_gradient("total");')
        path.write_text(s)
    path=BACK/'line_search/line_searcher.cu';s=path.read_text()
    if '// PSQ positive-infinity trial rejection' not in s:
        s='#include <finite_element/constitutions/psq_trial_energy.h> // PSQ positive-infinity trial rejection\n#include <finite_element/constitutions/psq_proxy.h>\n#include <cstdlib>\n'+s
        s=replace(s,'    auto reporter_energies = span{m_energy_values}.first(reporter_count);',
            '''    const char* inf_env=std::getenv("UIPC_PSQ_REJECT_INF_TRIAL");
    const bool allow_psq_inf=inf_env && std::string(inf_env)=="1" && psq_replaces_bending();
    bool saw_rejectable_inf=false;
    auto reporter_energies = span{m_energy_values}.first(reporter_count);''')
        s=replace(s,'        UIPC_ASSERT(!std::isnan(energy) && std::isfinite(energy),',
            '''        const bool rejectable=psq_trial_energy::rejectable_positive_infinity(
            energy,is_initial,std::string(reporter->name()).find("FEMLineSearchReporter")!=std::string::npos,allow_psq_inf);
        saw_rejectable_inf=saw_rejectable_inf || rejectable;
        UIPC_ASSERT(std::isfinite(energy) || rejectable,''')
        s=replace(s,'    UIPC_ASSERT(!std::isnan(total_energy) && std::isfinite(total_energy),',
            '    UIPC_ASSERT(std::isfinite(total_energy) || (saw_rejectable_inf && std::isinf(total_energy) && total_energy>0),')
        path.write_text(s)
    path=BACK/'engine/advance_ipc.cu';s=path.read_text()
    if '// PSQ reject nonfinite trial before convergence acceptance' not in s:
        s=replace(s,'    bool forced_frame = false;',
            '''    const char* inf_env=std::getenv("UIPC_PSQ_REJECT_INF_TRIAL");
    const bool reject_inf_trials=inf_env && std::string(inf_env)=="1" && psq_replaces_bending();
    bool forced_frame = false;''')
        s=replace(s,'                        last_E  = E;',
            '''                        last_E  = E;
                        // PSQ reject nonfinite trial before convergence acceptance
                        if(reject_inf_trials && std::isinf(E) && E>0)
                        {
                            logger::warn("PSQ_REJECT_INF_TRIAL frame={} newton={} search={} alpha={:.17g}",m_current_frame,newton_iter,line_search_iter,alpha);
                            alpha /= 2;
                            continue;
                        }''')
        s=replace(s,'#include <finite_element/constitutions/psq_preview_stop.inl>',
            '''                    if(reject_inf_trials && !std::isfinite(last_E))
                    {
                        compute_energy(0.0);
                        throw SimEngineException("PSQ trial energy stayed nonfinite after backtracking; restored current iterate");
                    }
#include <finite_element/constitutions/psq_preview_stop.inl>''')
        path.write_text(s)
    # Separate PSQ Algorithm 1 small-alpha stop from proven non-descent.
    path=BACK/'sim_engine.h';s=path.read_text()
    if 'm_frame_preview_small_step' not in s:
        s=replace(s,'    bool  m_frame_preview_non_descent      = false;',
            '    bool  m_frame_preview_non_descent      = false;\n    bool  m_frame_preview_small_step       = false;')
        path.write_text(s)
    path=BACK/'engine/sim_engine.cu';s=path.read_text()
    if 'm_frame_preview_small_step' not in s:
        s=replace(s,'    m_frame_preview_non_descent      = false;',
            '    m_frame_preview_non_descent      = false;\n    m_frame_preview_small_step       = false;')
        s=replace(s,'    stats["preview_non_descent"] = m_frame_preview_non_descent;',
            '    stats["preview_non_descent"] = m_frame_preview_non_descent;\n    stats["preview_small_step"] = m_frame_preview_small_step;')
        s=replace(s,'m_frame_preview_non_descent ? "preview_non_descent" : (m_frame_converged',
            'm_frame_preview_small_step ? "preview_small_step" : m_frame_preview_non_descent ? "preview_non_descent" : (m_frame_converged')
        path.write_text(s)
    path=BACK/'engine/advance_ipc.cu';s=path.read_text()
    if 'm_frame_preview_small_step' not in s:
        s=replace(s,'if(forced_frame || m_frame_preview_non_descent) return;',
            'if(forced_frame || m_frame_preview_non_descent || m_frame_preview_small_step) return;')
        path.write_text(s)
    # Copy package contents rather than nesting python/python on incremental builds.
    for path in [SRC/'src/pybind/xmake.lua',HERE/'build_support/pybind_xmake.lua']:
        s=path.read_text().replace('os.cp(python_source_dir, python_build_dir, {','os.cp(path.join(python_source_dir, "*"), python_build_dir, {')
        path.write_text(s)
    shutil.copytree(SRC/'python/src/uipc',SRC/'build/python/src/uipc',dirs_exist_ok=True)
    # Match xmake's actual dynamic dependency; do not mix in wheel libraries.
    import os
    spdlog=Path(os.environ['LOCALAPPDATA'])/'.xmake/packages/s/spdlog/v1.17.0/2b88d46e92f5490a92a28ec98fee3035/bin/spdlog.dll'
    shutil.copy2(spdlog,SRC/'build/python/src/uipc/_native/spdlog.dll')

if __name__=='__main__':main()

