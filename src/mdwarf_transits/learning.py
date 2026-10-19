"""Original TensorFlow training update, inference, and checkpoint serialization."""
import hashlib
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from .model import source_loss
from .io import write, digest

def configure(r):
    if tf.__version__!=r['runtime']['tensorflow'] or np.__version__!=r['runtime']['numpy']:raise ValueError('Unqualified runtime')
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(r['intra_threads'])
    tf.config.threading.set_inter_op_parallelism_threads(r['inter_threads'])

def updater(model,optimizer):
    loss=source_loss()
    @tf.function
    def update(x,y):
        with tf.GradientTape() as tape:l=loss(y,model(x,training=True))
        gradients=tape.gradient(l,model.trainable_variables)
        tf.debugging.assert_all_finite(l,'Nonfinite loss')
        for g in gradients:
            if g is None:raise ValueError('Disconnected gradient')
            tf.debugging.assert_all_finite(g,'Nonfinite gradient')
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return l
    return update

def predict(model,x,batch_size):
    z=np.concatenate([model.logits(x[i:i+batch_size],training=False).numpy().reshape(-1) for i in range(0,len(x),batch_size)])
    if not np.isfinite(z).all():raise ValueError('Nonfinite logits')
    return z

def array_hash(values):
    h=hashlib.sha256()
    for a in values:h.update(np.asarray(a).tobytes())
    return h.hexdigest()

def save_checkpoint(folder,model,opt,sampler,step):
    folder=Path(folder);tmp=folder.with_name(folder.name+'.partial');tmp.mkdir(exist_ok=False)
    model.save_weights(str(tmp/'ordinary.weights.h5'))
    tf.train.Checkpoint(model=model,optimizer=opt).write(str(tmp/'state'))
    np.savez(str(tmp/'sampler.npz'),sample_keys=sampler.sample_rng.get_state()[1],augmentation_keys=sampler.augmentation_rng.get_state()[1],
        x=np.stack([v[0] for v in sampler.buffer]) if sampler.buffer else np.empty((0,1000,1),np.float32),
        y=np.array([v[1] for v in sampler.buffer]),indices=np.array([v[2] for v in sampler.buffer]),draw_ids=np.array([v[3] for v in sampler.buffer]))
    def rng_tail(rng):
        s=rng.get_state();return {'algorithm':s[0],'position':s[2],'has_gauss':s[3],'cached_gaussian':s[4]}
    write(tmp/'state.json',{'step':step,'optimizer_iteration':int(opt.iterations.numpy()),'draws':sampler.draws,'exposures':sampler.exposures,
       'sampler_rng':rng_tail(sampler.sample_rng),'augmentation_rng':rng_tail(sampler.augmentation_rng),
       'resume':'Fresh restart required after termination; TensorFlow dropout stream is not restored.'})
    write(tmp/'completion.json',{'passed':True,'hashes':{p.name:digest(p) for p in tmp.iterdir() if p.is_file()}})
    os.rename(str(tmp),str(folder))
