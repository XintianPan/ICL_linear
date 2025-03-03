import torch
import numpy as np
from typing import Any, Dict, Optional

class DataMethod:
    def __init__(self, dict: Dict = None):
        # Initialize the DataMethod class with an optional dictionary of parameters.
        self.dict = dict

    def __generatedata__(self, **kwargs) -> Any:
        """
        This method generates synthetic data based on specified sequence length and dimension.

        Parameters:
            kwargs: Additional keyword arguments (not used here but allows for flexible method signature).

        Returns:
            torch.Tensor: Generated data tensor with shape (seq_len, dim).
        """
        # Retrieve the sequence length and dimension from the dictionary (with default values if not provided).
        seq_len = self.dict.get("seq_length", 100)
        dim = self.dict.get("dimension", 10)
        # Generate a tensor with random values sampled from a standard normal distribution.
        x = torch.randn(seq_len, dim)
        return x

    def __transform__(self, x: Any, **kwargs) -> Any:
        """
        This method transforms input data for training, validation, or testing purposes.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: The input tensor excluding the last time step (shape: batch_size, seq_len - 1, dim).
                - y: The corresponding target tensor excluding the first time step (shape: batch_size, seq_len - 1, dim).
        """
        # Create a target tensor by removing the first element along the sequence dimension.
        y = x[..., 1:, :].clone()
        # Trim the input tensor to remove the last element along the sequence dimension.
        x = x[..., :-1, :]
        return x, y


def generate_covariance_matrix(d, rho=0.5):
    """
    Generate a d by d covariance matrix where each entry is given by rho^|i-j|,
    and return the covariance matrix along with its eigenvalues.

    Parameters:
        d (int): Dimension of the covariance matrix.
        rho (float): Correlation coefficient, where -1 < rho < 1.

    Returns:
        tuple: A tuple containing the covariance matrix (numpy.ndarray) and its eigenvalues (numpy.ndarray).
    """
    if not (-1 < rho < 1):
        raise ValueError("rho must be between -1 and 1 (exclusive).")

    # Create the covariance matrix
    covariance_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            covariance_matrix[i, j] = rho ** abs(i - j)

    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)

    return covariance_matrix, eigenvalues

class LinearReg(DataMethod):
    """
    This class generates data for a linear regression task based on specific parameters, 
    such as data size, sequence length, noise level, and condition number of the covariance matrix.
    """

    def __init__(self, dict: Dict = None):
        """
        Initialize the LinearReg class with a set of parameters.

        Parameters:
            dict (Dict): Dictionary containing parameters for data generation.
        """
        # Call the parent class initializer.
        super().__init__(dict)
        # Extract parameters for data generation.
        self.L = dict['L']  # Sequence length
        self.dx = dict['dx']  # Input dimension
        self.dy = dict['dy']  # Output dimension
        self.noise_std = dict['noise_std']  # Standard deviation of the noise
        self.number_of_samples = dict['number_of_samples']  # Number of data samples
        

    def __generatedata__(self, **kwargs) -> Any:
        """
        Generate linear regression data.

        Parameters:
            kwargs: Additional keyword arguments (not used here).

        Returns:
            Tuple: Generated data tensors (z_q, z, y_q).
        """
        # Generate input data with shape (n, L, dx).
        x = torch.randn(self.number_of_samples, self.L, self.dx)

        # Generate query data (single time-step data) with shape (n, 1, dx).
        x_q = torch.randn(self.number_of_samples, 1, self.dx)

        # Generate regression coefficients (beta) with shape (n, dx, dy).
        beta = torch.randn(self.number_of_samples, self.dx, self.dy) * torch.sqrt(torch.tensor(1/self.dx))
        
        # Generate target output data y with shape (n, L, dy) using x and beta.
        y = torch.einsum('nlx,nxy->nly', x, beta)
        # Add Gaussian noise to the output y.
        #y += math.sqrt(self.dx) * self.noise_std * torch.randn(self.number_of_samples, self.L, self.dy)
        y += self.noise_std * torch.randn(self.number_of_samples, self.L, self.dy)
        # Generate output data for query points y_q with shape (n, 1, dy).
        y_q = torch.einsum('nlx,nxy->nly', x_q, beta)
        y_q += self.noise_std * torch.randn(self.number_of_samples, 1, self.dy)

        # Concatenate x and y to form a combined tensor z for training purposes.
        z = torch.cat([x, y], dim=2)
        # Concatenate x_q with a zero-filled tensor to form z_q for query purposes.
        z_q = torch.cat([x_q, torch.zeros_like(y_q)], dim=2)
        return z_q.squeeze(0), z.squeeze(0), y_q

    def __transform__(self, x: Any, zero_index: Optional[int] = None, **kwargs) -> Any:
        """
        Transform the data for training, validation, and testing.

        Parameters:
            x (Any): Input data tensor.
            zero_index (Optional[int]): Index to set to zero in the data (if provided).

        Returns:
            Tuple: Transformed input and target tensors.
        """
        # Extract the last dimension of the data as the target output.
        y = x[..., :, -1].clone()

        # Optionally zero out a specified index.
        if zero_index is not None:
            x[..., zero_index, -1] = 0

        return x, y