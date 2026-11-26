"""Isolated advance-loop overlay; all PCG/assembly objects remain unchanged."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/flag_L3_fix_20261118'
PARENT = BASE / 'bunny_ordered_pcg_build'
BUILD = BASE / 'bunny_event_fallback_build'
RUNTIME = BASE / 'bunny_event_fallback_runtime/src'


def main():
    BUILD.mkdir(exist_ok=False)
    shutil.copytree(BASE / 'bunny_ordered_pcg_runtime/src', RUNTIME)
    for p in PARENT.glob('*.obj'):
        shutil.copy2(p, BUILD / p.name)
    original = BASE / 'bunny_precision_build/advance_ipc.cu'
    code = original.read_text()
    code = code.replace('IndexT flag_rescue_iteration=-1;', '''IndexT flag_rescue_iteration=-1;
            bool pilot_exact_active=false;
            IndexT pilot_exact_start=-1;
            const bool pilot_enabled=std::getenv("BUNNY_EVENT_FALLBACK")
                && std::string(std::getenv("BUNNY_EVENT_FALLBACK"))=="1";''')
    code = code.replace('m_global_linear_system->solve();', '''if(pilot_exact_active) _putenv_s("UIPC_PSQ_EXACT_ONCE","1");
                    logger::info("BUNNY_EVENT_MODE frame={} newton={} exact={}",m_current_frame,newton_iter,pilot_exact_active);
                    m_global_linear_system->solve();''')
    code = code.replace('bool  converged        = convergence_check(newton_iter);', '''bool  converged = convergence_check(newton_iter);
                    // Require at least three exact-gradient directions after switching.
                    if(pilot_exact_active && newton_iter-pilot_exact_start<2) converged=false;''')
    sources = [original]
    for filename, directory, reason in [
        ('psq_preview_stop.inl', ROOT / 'native/flag_l3_repair', 'finite_difference'),
        ('bunny_preview_derivative.inl', ROOT / 'native/libuipc', 'analytic'),
        ('bunny_representable_search.inl', ROOT / 'native/libuipc', 'representable')]:
        p = directory / filename; sources.append(p)
        body = p.read_text()
        # Disable preview checks after switching; ordinary safeguarded search remains.
        body = body.replace('if(preview_requested && !converged', 'if(!pilot_exact_active && preview_requested && !converged')
        body = body.replace('if(line_search_iter >=', 'if(!pilot_exact_active && line_search_iter >=')
        hook = '''if(pilot_enabled && !pilot_exact_active) {
            const Float pilot_retained=compute_energy(0.0);
            const Float pilot_noise=64*std::numeric_limits<Float>::epsilon()*std::max(std::abs(E0),Float(1e-30));
            if(!std::isfinite(pilot_retained) || std::abs(pilot_retained-E0)>pilot_noise)
                throw SimEngineException("Event fallback failed to restore current iterate");
            pilot_exact_active=true; pilot_exact_start=newton_iter+1;
            m_frame_hit_line_search_limit=false;
            m_frame_last_line_search_alpha=0;
            logger::info("BUNNY_EVENT_TRIGGER frame={} newton={} reason=REASON E0={:.17g} retained={:.17g}",
                         m_current_frame,newton_iter,E0,pilot_retained);
            continue;
        }
        '''.replace('REASON', reason)
        assert body.count('m_frame_preview_non_descent = true;') == 1
        body = body.replace('m_frame_preview_non_descent = true;', hook + 'm_frame_preview_non_descent = true;')
        # Include directly so quoted includes cannot resolve to the old overlay.
        token = '#include "' + filename + '"'
        assert code.count(token) == 1
        code = code.replace(token, body)
    patched = BUILD / 'advance_ipc.cu'; patched.write_text(code)
    env = os.environ.copy(); env.update(json.loads((BASE / 'build_env.json').read_text())['env'])
    commands = json.loads((BASE / 'bunny_precision_build/commands.json').read_text())
    compile_cmd = next(c for c in commands if str(original) in c)
    compile_cmd = [s.replace(str(original), str(patched)).replace(
        str(original.with_suffix('.cu.obj')), str(BUILD / 'advance_ipc.cu.obj')) for s in compile_cmd]
    # The source and object names overlap; map the full old build path as well.
    compile_cmd = [s.replace(str(original.parent), str(BUILD)) for s in compile_cmd]
    parent_commands = json.loads((PARENT / 'commands.json').read_text())
    commands = [compile_cmd]
    for c in parent_commands[-2:]:
        commands.append([s.replace(str(PARENT), str(BUILD)).replace(
            str(BASE / 'bunny_ordered_pcg_runtime/src'), str(RUNTIME)) for s in c])
    (BUILD / 'commands.json').write_text(json.dumps(commands, indent=2))
    for i, c in enumerate(commands):
        print('Build stage', i, flush=True)
        if i == 0:
            subprocess.run(c, cwd=ROOT / 'external/libuipc-psq-src', env=env, check=True)
        else:
            rsp = BUILD / f'stage{i}.rsp'
            if 'nvcc' in c[0]:
                rsp.write_text('\n'.join('"' + s.replace('\\', '/') + '"' for s in c[1:]))
                cmd = [c[0], '-optf', str(rsp)]
            else:
                rsp.write_text(subprocess.list2cmdline(c[1:])); cmd = [c[0], '@' + str(rsp)]
            subprocess.run(cmd, cwd=ROOT / 'external/libuipc-psq-src', env=env, check=True)
    manifest = json.loads((PARENT / 'provenance.json').read_text())
    manifest.update(dll_sha256=hashlib.sha256((RUNTIME / 'uipc/_native/uipc_backend_cuda.dll').read_bytes()).hexdigest(),
        runtime=str(RUNTIME), reference_dll_sha256=manifest['dll_sha256'],
        event_fallback_control=True, changed_unit='advance_ipc.cu', shared_sources_written=False)
    for p in [*sources, patched, Path(__file__)]:
        manifest['source_hashes'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in PARENT.glob('*.obj'):
        if p.name not in ('advance_ipc.cu.obj', 'gpucode.obj'):
            assert p.read_bytes() == (BUILD / p.name).read_bytes()
    (BUILD / 'provenance.json').write_text(json.dumps(manifest, indent=2))
    print('Built', RUNTIME, flush=True)


if __name__ == '__main__':
    main()
