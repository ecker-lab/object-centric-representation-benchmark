object_room_variant = dict(
    op3_args=dict(
        refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
        decoder_model_type="reg",  # reg, reg_no_share
        dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
        sto_repsize=64,
        det_repsize=64,
        extra_args=dict(
            beta=1e-2,
            deterministic_sampling=False
        ),
        K=8
    ),
    schedule_args=dict(  # Arguments for TrainingScheduler
        seed_steps=4,
        T=5,  # Max number of steps into the future we want to go or max length of a schedule
        schedule_type='rprp',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
        loss_type='iodine_dynamics',
        r_steps=2,
    ),
    training_args=dict(  # Arguments for OP3Trainer
        batch_size=16,  # Change to appropriate constant based off dataset size
        lr=3e-4,
    ),
    num_epochs=300,
    save_period=5,
    dataparallel=True,
    path='ocrb/data/datasets',
    ckpt_dir='ocrb/op3/ckpts',
    num_workers=1,
    n_steps=10,
    debug=False,
    split=True
)

mot_sprite_variant = dict(
    op3_args=dict(
        refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
        decoder_model_type="reg",  # reg, reg_no_share
        dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
        sto_repsize=64,
        det_repsize=64,
        extra_args=dict(
            beta=1e-2,
            deterministic_sampling=False
        ),
        K=5
    ),
    schedule_args=dict(  # Arguments for TrainingScheduler
        seed_steps=4,
        T=5,  # Max number of steps into the future we want to go or max length of a schedule
        schedule_type='rprp',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
        loss_type='iodine_dynamics',
        r_steps=2,
    ),
    training_args=dict(  # Arguments for OP3Trainer
        batch_size=16,  # Change to appropriate constant based off dataset size
        lr=1e-4
    ),
    num_epochs=300,
    save_period=5,
    dataparallel=True,
    path='ocrb/data/datasets',
    ckpt_dir='ocrb/op3/ckpts',
    num_workers=1,
    n_steps=10,
    debug=False,
    split=True
)

multidsprites_videos_variant = dict(
    op3_args=dict(
        refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
        decoder_model_type="reg",  # reg, reg_no_share
        dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
        sto_repsize=64,
        det_repsize=64,
        extra_args=dict(
            beta=1e-2,
            deterministic_sampling=False
        ),
        K=6
    ),
    schedule_args=dict(  # Arguments for TrainingScheduler
        seed_steps=4,
        T=5,  # Max number of steps into the future we want to go or max length of a schedule
        schedule_type='rprp',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
        loss_type='iodine_dynamics',
        r_steps=2,
    ),
    training_args=dict(  # Arguments for OP3Trainer
        batch_size=16,  # Change to appropriate constant based off dataset size
        lr=3e-4,
    ),
    num_epochs=300,
    save_period=5,
    dataparallel=True,
    path='ocrb/data/datasets',
    ckpt_dir='ocrb/op3/ckpts',
    num_workers=1,
    n_steps=10,
    debug=False,
    split=True
)