import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import torch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import DirichletClassificationLikelihood


import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO


class GPClassifierModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dim):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPClassifierModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FixedMeanGPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=False)
        super(FixedMeanGPClassificationModel, self).__init__(variational_strategy)

        # Fixed mean function
        self.mean_module = gpytorch.means.ZeroMean()

        # RBF Kernel with learnable lengthscale
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module.outputscale = 2.0  # Fixed outputscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



class EnhancedDirichletGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, embedding_size=512, reduced_size=128, num_classes=2):
        super(EnhancedDirichletGPModel, self).__init__(train_x, train_y, likelihood)

        # The representation layer to reduce dimensionality
        self.representation_layer = torch.nn.Linear(embedding_size, reduced_size)

        # Batch normalization layer
        self.batch_norm = torch.nn.BatchNorm1d(reduced_size)

        # The GP layer
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # Classifier layer, assuming usage of Dirichlet likelihood for classification
        self.classifier = torch.nn.Linear(reduced_size, num_classes)

    def forward(self, x):
        # Pass through the representation and normalization layers
        representation = self.representation_layer(x)
        representation = self.batch_norm(representation)

        mean_x = self.mean_module(representation)
        covar_x = self.covar_module(representation)
        gp_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return gp_pred



# class DirichletGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, num_classes):
#         super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes])),
#             batch_shape=torch.Size([num_classes]),
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#         # print("Forward pass outputscales:", self.covar_module.outputscale)
#         return mvn


class DirichletGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)), #linear
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class DirichletGPModel(ExactGP):
#     def __init__(self, train_x, train_y, likelihood, num_classes):
#         super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
#         self.covar_module = ScaleKernel(
#             RBFKernel(batch_shape=torch.Size((num_classes,))),
#             batch_shape=torch.Size((num_classes,)),
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        print(x)
        mean_x = self.mean_module(x)
        print(mean_x)
        covar_x = self.covar_module(x)
        print(covar_x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        print(latent_pred)
        return latent_pred


#
class GPModelLinearMean(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModelLinearMean, self).__init__(variational_strategy)
        # Use a linear mean function
        self.mean_module = gpytorch.means.LinearMean(input_size=inducing_points.size(-1))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelConstantMean(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModelConstantMean, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return mvn