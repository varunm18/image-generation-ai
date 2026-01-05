import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from torchvision.utils import make_grid
import imageio.v2 as imageio


def plot_losses(losses, window=1, label="Loss", ax=None):
    weights = np.ones(window) / window

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(np.convolve(losses, weights, mode="valid"), label=label)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()

    return fig, ax


def gif_from_images(images_dir, save_dir, fps=10, file_name="training"):
    images = []

    files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

    for file in files:
        img_path = os.path.join(images_dir, file)
        images.append(imageio.imread(img_path))

    imageio.mimsave(f"{save_dir}/{file_name}.gif", images, fps=fps)


class AutoEncoderBase(ABC):
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        train_dataset,
        test_dataset,
        device=torch.device("cpu"),
    ):
        self.device = device
        self.model = model.to(device)

        self.optimizer = optimizer

        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    @abstractmethod
    def get_loss(self, input):
        pass

    def train(self, epoch):
        self.model.train()
        losses = []

        with tqdm(self.train_loader, unit="batch", desc=f"Epoch #{epoch + 1}") as pbar:
            for input, _ in pbar:
                input = input.to(self.device)

                loss = self.get_loss(input)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

                pbar.set_postfix({"Avg Loss": f"{sum(losses) / len(losses):.4f}"})

        return losses

    def test(self):
        self.model.eval()

        losses = []

        with torch.no_grad():
            with tqdm(self.test_loader, unit="batch", desc=f"Test Set") as pbar:
                for input, _ in pbar:
                    input = input.to(self.device)

                    loss = self.get_loss(input)

                    losses.append(loss.item())

                    pbar.set_postfix({"Avg Loss": f"{sum(losses) / len(losses):.4f}"})

        return losses


class AutoEncoder(AutoEncoderBase):
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        train_dataset,
        test_dataset,
        device=torch.device("cpu"),
    ):
        super().__init__(
            model, optimizer, batch_size, train_dataset, test_dataset, device
        )

        self.loss_func = nn.MSELoss()

    def get_loss(self, images):
        reconstructed = self.model(images)
        return self.loss_func(reconstructed, images)

    def get_batch(self):
        with torch.no_grad():
            images, _ = next(
                iter(
                    DataLoader(
                        self.test_loader.dataset,
                        batch_size=self.test_loader.batch_size,
                        shuffle=True,
                    )
                )
            )

            images = images.to(self.device)

            reconstructed = self.model(images)

        return images, reconstructed


class VariationalAutoEncoder(AutoEncoderBase):
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        train_dataset,
        test_dataset,
        device=torch.device("cpu"),
        beta=1,
    ):
        super().__init__(
            model, optimizer, batch_size, train_dataset, test_dataset, device
        )

        self.beta = beta

    def get_loss(self, images):
        reconstructed, mu, logvar = self.model(images)
        recon_loss = nn.functional.mse_loss(reconstructed, images, reduction="sum")

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def get_batch(self):
        with torch.no_grad():
            images, _ = next(
                iter(
                    DataLoader(
                        self.test_loader.dataset,
                        batch_size=self.test_loader.batch_size,
                        shuffle=True,
                    )
                )
            )

            images = images.to(self.device)

            reconstructed, _, _ = self.model(images)

        return images, reconstructed

    def sample(self, n=5):
        with torch.no_grad():
            z = torch.randn(n, self.model.latent_dim).to(self.device)
            samples = self.model.decoder(z)

        return samples


class GAN:
    def __init__(
        self,
        discriminator,
        generator,
        d_optimizer,
        g_optimizer,
        batch_size,
        latent_dim,
        save_dir,
        dataset,
        device=torch.device("cpu"),
    ):
        self.device = device
        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.batch_size = batch_size
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.latent_dim = latent_dim
        self.fixed_noise = torch.randn(25, latent_dim).to(device)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.loss_fn = nn.BCELoss()

    def save_generated_images(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            gen_imgs = self.generator(self.fixed_noise)

            # [-1,1] from tanh -> [0,1]
            gen_imgs = (gen_imgs + 1) / 2
            gen_imgs = gen_imgs.view(-1, 1, 28, 28)

            grid = make_grid(gen_imgs, nrow=5)

        grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.imshow(grid, cmap="gray")
        plt.axis("off")

        plt.text(
            0.5,
            -0.05,
            f"Epoch {epoch}",
            fontsize=12,
            ha="center",
            transform=plt.gca().transAxes,
        )

        plt.savefig(f"{self.save_dir}/epoch_{epoch:03d}.png", bbox_inches="tight")
        plt.close()

    def train(self, epoch):
        self.discriminator.train()
        self.generator.train()

        epoch_g_loss, epoch_d_loss = [], []

        with tqdm(self.loader, unit="batch", desc=f"Epoch #{epoch + 1}") as pbar:
            for real_imgs, _ in pbar:
                real_imgs = real_imgs.to(self.device)

                valid = torch.ones(real_imgs.size(0), 1).to(self.device)
                fake = torch.zeros(real_imgs.size(0), 1).to(self.device)

                z = torch.randn(real_imgs.size(0), self.latent_dim).to(self.device)
                gen_imgs = self.generator(z)

                g_loss = self.loss_fn(self.discriminator(gen_imgs), valid)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                d_loss = self.loss_fn(self.discriminator(real_imgs), valid)
                d_loss += self.loss_fn(self.discriminator(gen_imgs.detach()), fake)

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                epoch_g_loss.append(g_loss.item())
                epoch_d_loss.append(d_loss.item())

                pbar.set_postfix(
                    {
                        "Avg Gen. Loss": f"{sum(epoch_g_loss) / len(epoch_g_loss):.4f}",
                        "Avg Disc. Loss": f"{sum(epoch_d_loss) / len(epoch_d_loss):.4f}",
                    }
                )

        self.save_generated_images(epoch + 1)

        return epoch_g_loss, epoch_d_loss

    def plot_samples(self, n=5):
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(n * n, self.latent_dim).to(self.device)
            samples = self.generator(z)

        samples = samples.view(-1, 28, 28).cpu().detach()

        fig = plt.figure(figsize=(n, n))

        for i in range(n * n):
            plt.subplot(n, n, i + 1)
            plt.imshow(samples[i], cmap="gray")
            plt.axis("off")

        plt.suptitle(f"MNIST GAN Samples ({n}x{n})")
        plt.tight_layout()

        return fig
