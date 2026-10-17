# ========================================================================
# The copyrights of the software are owned by Duke University.
# Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
# Department of Computer Science, Trinity College of Arts & Sciences, Duke
# University.
# ========================================================================
# DON'T MODIFY. THESE DEFINE THE MODEL ARCHITECTURE IN MMCV STYLE
modelname = 'fulltilt'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
transformer_activation = 'relu'
num_queries = 900

# TRAINING HYPERPARAMETERS
# OPTIMIZATION RELATED
batch_size = 1
epochs = 500
weight_decay = 0.0001
lr = 0.0001
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_mult = 1e-05
lr_linear_proj_names = ['sampling_offsets']
onecyclelr = False
multi_step_lr = False
lr_drop_list = [500, 1000]
lr_drop = 500
clip_max_norm = 0.1

# AUGMENTATION RELATED
strong_aug = True
random_order_aug = True

# PRETRAINED WEIGHTS RELATED
backbone_freeze_keywords = None  # TODO: should be removed
freeze_keywords = []

# CHECKPOINT RELATED
save_checkpoint_interval = 1
eval_interval = 5

# MATCHER RELATED
matcher_type = 'HungarianMatcher'
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

# LOSS RELATED
aux_loss = True
dn_number = 100
dn_box_noise_scale = 0.4
cls_loss_coef = 2.0  # NOTE: actually not used
bbox_loss_coef = 5.0  # NOTE: actually not used
giou_loss_coef = 2.0  # NOTE: actually not used
bbox_3d_loss_coef = 10.0
ce_3d_loss_coef = 2.0
no_interm_box_loss = False
interm_loss_coef = 1.0
focal_alpha = 0.25
focal_gamma = 2.0
num_select = 900  # for evaluation in training

# OTHERS
# max number of unique classes (can be set lower or higher as this is not related to model
# architecture)
max_unique_classes = 256
