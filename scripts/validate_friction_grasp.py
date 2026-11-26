"""Compare matched mu=1 / mu=0 runs; do not infer grasp from a pretty video."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validate(held_root,slip_root,output):
    held_cfg=json.loads((held_root/'experiment.json').read_text())
    slip_cfg=json.loads((slip_root/'experiment.json').read_text())
    assert held_cfg['completed'] and slip_cfg['completed']
    for key,value in held_cfg['parameters'].items():
        if key not in ('friction','output'):
            assert value==slip_cfg['parameters'][key],key
    assert held_cfg['parameters']['friction']==1 and slip_cfg['parameters']['friction']==0
    assert held_cfg['mesh_hashes']==slip_cfg['mesh_hashes']
    results=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for level in range(held_cfg['parameters']['levels']):
        a=held_root/f'direct_L{level}';b=slip_root/f'direct_L{level}'
        x=np.load(a/'positions.npy');y=np.load(b/'positions.npy');poses=np.load(a/'robot.npy')
        top=x[0,:,2]>.065
        held=np.mean(x[:,top,2]-x[0,top,2],axis=1)
        slip=np.mean(y[:,top,2]-y[0,top,2],axis=1)
        jaw=np.mean(poses[:,:,2,3]-poses[0,:,2,3],axis=1)
        assert held[-1]>.01 and abs(held[-1]-jaw[-1])<.002
        assert slip[-1]<-.2
        results.append({'level':level,'held_top_displacement_m':float(held[-1]),
                        'zero_friction_top_displacement_m':float(slip[-1]),'jaw_displacement_m':float(jaw[-1]),
                        'passed':True})
        t=np.arange(len(x))*held_cfg['dt']
        axes[0].plot(t,slip,label=f'mu=0, L{level}')
        axes[1].plot(t,held*1000,label=f'mu=1, L{level}')
    axes[1].plot(t,jaw*1000,'k--',label='actual jaw')
    axes[0].set_title('Without friction: cloth slips');axes[0].set_ylabel('Top cloth displacement (m)')
    axes[1].set_title('With friction: cloth follows jaw');axes[1].set_ylabel('Top cloth displacement (mm)')
    for ax in axes:ax.grid(alpha=.25);ax.legend();ax.set_xlabel('Time (s)')
    output.mkdir(exist_ok=True,parents=True)
    fig.savefig(output/'friction_control.png',dpi=150);plt.close(fig)
    record={'held_run':str(held_root),'zero_friction_run':str(slip_root),'matched_parameters':True,
            'scope':'synthetic physical jaw/cloth integration test, not the full Scene527 fold','results':results}
    (output/'friction_control.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    validate(Path('output/pd_grip_mu1_k1e4'),Path('output/pd_grip_mu0_k1e4'),Path('output/pd_framework_validation'))
