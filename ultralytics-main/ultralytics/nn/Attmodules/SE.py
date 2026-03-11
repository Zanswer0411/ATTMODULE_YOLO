class SE(nn.Module):

    def __init__(self, channels: int , reduction: int = 16):

        super().__init__()
        assert channels >0
        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = self.gap(x)
        s = self.fc2(self.act(self.fc1(z)))
        s = self.sigmoid(s)
        out = x * s

        return out

if __name__ == "__main__":
    x = torch.randn(2, 64, 80, 80)
    se = SE(channels=64, reduction=16)
    y = se(x)
    print("x:", x.shape, "y:", y.shape)