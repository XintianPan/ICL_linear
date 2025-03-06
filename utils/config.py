class Config:
    def __init__(
        self, 
        n_layers=2, 
        n_head=1,
        n_out =2,
        d=5,
        dx1=3,
        dx2=4,
        weight_tying = False,
        L=None, 
        method='SGD', 
        learning_rate=0.001, 
        batch_size=128, 
        optimizer_params=None, 
        training_steps=100001, 
        dynamic_log_every_step=1000,
        qk_log_every_step=40000,
        loss_log_every_step=1000, 
        validation_every=1000, 
        print_loss_every=50000, 
        noise_std=0.01
    ):
        """
        Initialize the configuration for an experiment with default values.

        Parameters:
            n_layers (int): Number of MultiHeadAttention layers. Default is 2.
            n_head (int): Number of attention heads. Default is 1.
            d (int): Embedding dimension (embedding size will be d + 1). Default is 10.
            L (int, optional): Sequence length for data generation. Default is 8 * d.
            method (str): Optimization method ('SGD', 'Adam', 'AdamW'). Default is 'SGD'.
            learning_rate (float): Learning rate for optimization. Default is 0.001.
            batch_size (int): Batch size for training. Default is 128.
            optimizer_params (dict): Additional parameters for the optimizer. Default is None.
            training_steps (int): Total number of training steps. Default is 100001.
            dynamic_log_every_step (int): Interval for logging QK and OV dynamics. Default is 1000.
            qk_log_every_step (int): Interval for logging QK matrices. Default is 40000
            loss_log_every_step (int): Interval for logging loss. Default is 1000.
            validation_every (int): Interval for running validation. Default is 1000.
            print_loss_every (int): Interval for printing loss during validation. Default is 5000.
            noise_std (float): Standard deviation for noise in data generation. Default is 0.01.
        """
        self.n_layers = n_layers
        self.n_head = n_head
        self.d = d
        self.dx1 = dx1
        self.dx2 = dx2
        self.n_embd = d + 2  # Embedding dimension for the model is d + 1
        self.n_out = n_out
        self.weight_tying = weight_tying
        self.L = L if L is not None else 8 * d  # Default is 8 * d
        self.method = method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.noise_std = noise_std
        
        self.optimizer_params = optimizer_params if optimizer_params is not None else {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0}
        self.training_steps = training_steps
        self.dynamic_log_every_step = dynamic_log_every_step
        self.qk_log_every_step = qk_log_every_step
        self.loss_log_every_step = loss_log_every_step
        self.validation_every = validation_every
        self.print_loss_every = print_loss_every

    def __repr__(self):
        return (f"Config(\n"
                f"  n_layers: {self.n_layers}\n"
                f"  n_head: {self.n_head}\n"
                f"  d: {self.d}\n"
                f"  L: {self.L}\n"
                f"  method: {self.method}\n"
                f"  learning_rate: {self.learning_rate}\n"
                f"  batch_size: {self.batch_size}\n"
                f"  noise_std: {self.noise_std}\n"
                f"  optimizer_params: {self.optimizer_params}\n"
                f"  training_steps: {self.training_steps}\n"
                f"  dynamic_log_every_step: {self.dynamic_log_every_step}\n"
                f"  qk_log_every_step: {self.qk_log_every_step}\n"
                f"  loss_log_every_step: {self.loss_log_every_step}\n"
                f"  validation_every: {self.validation_every}\n"
                f"  print_loss_every: {self.print_loss_every}\n"
                f"  n_embd (d + 1): {self.n_embd}\n"
                f")")
    

class MultiTaskConfig:
    def __init__(
        self, 
        n_layers=2, 
        n_head=1,
        n_out =2,
        d=5,
        dx1=3,
        dx2=4,
        weight_tying = False,
        L=None, 
        method='SGD', 
        learning_rate=0.001, 
        batch_size=128, 
        optimizer_params=None, 
        training_steps=100001, 
        dynamic_log_every_step=1000,
        qk_log_every_step=40000,
        loss_log_every_step=1000, 
        validation_every=1000, 
        print_loss_every=50000, 
        noise_std=0.01
    ):
        """
        Initialize the configuration for an experiment with default values.

        Parameters:
            n_layers (int): Number of MultiHeadAttention layers. Default is 2.
            n_head (int): Number of attention heads. Default is 1.
            d (int): Embedding dimension (embedding size will be d + 1). Default is 10.
            L (int, optional): Sequence length for data generation. Default is 8 * d.
            method (str): Optimization method ('SGD', 'Adam', 'AdamW'). Default is 'SGD'.
            learning_rate (float): Learning rate for optimization. Default is 0.001.
            batch_size (int): Batch size for training. Default is 128.
            optimizer_params (dict): Additional parameters for the optimizer. Default is None.
            training_steps (int): Total number of training steps. Default is 100001.
            dynamic_log_every_step (int): Interval for logging QK and OV dynamics. Default is 1000.
            qk_log_every_step (int): Interval for logging QK matrices. Default is 40000
            loss_log_every_step (int): Interval for logging loss. Default is 1000.
            validation_every (int): Interval for running validation. Default is 1000.
            print_loss_every (int): Interval for printing loss during validation. Default is 5000.
            noise_std (float): Standard deviation for noise in data generation. Default is 0.01.
        """
        self.n_layers = n_layers
        self.n_head = n_head
        self.d = d
        self.dx1 = dx1
        self.dx2 = dx2
        self.n_embd = d + 2  # Embedding dimension for the model is d + 1
        self.n_out = n_out
        self.weight_tying = weight_tying
        self.L = L if L is not None else 8 * d  # Default is 8 * d
        self.method = method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.noise_std = noise_std
        
        self.optimizer_params = optimizer_params if optimizer_params is not None else {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0}
        self.training_steps = training_steps
        self.dynamic_log_every_step = dynamic_log_every_step
        self.qk_log_every_step = qk_log_every_step
        self.loss_log_every_step = loss_log_every_step
        self.validation_every = validation_every
        self.print_loss_every = print_loss_every

    def __repr__(self):
        return (f"Config(\n"
                f"  n_layers: {self.n_layers}\n"
                f"  n_head: {self.n_head}\n"
                f"  d: {self.d}\n"
                f"  L: {self.L}\n"
                f"  method: {self.method}\n"
                f"  learning_rate: {self.learning_rate}\n"
                f"  batch_size: {self.batch_size}\n"
                f"  noise_std: {self.noise_std}\n"
                f"  optimizer_params: {self.optimizer_params}\n"
                f"  training_steps: {self.training_steps}\n"
                f"  dynamic_log_every_step: {self.dynamic_log_every_step}\n"
                f"  qk_log_every_step: {self.qk_log_every_step}\n"
                f"  loss_log_every_step: {self.loss_log_every_step}\n"
                f"  validation_every: {self.validation_every}\n"
                f"  print_loss_every: {self.print_loss_every}\n"
                f"  n_embd (d + 1): {self.n_embd}\n"
                f")")

