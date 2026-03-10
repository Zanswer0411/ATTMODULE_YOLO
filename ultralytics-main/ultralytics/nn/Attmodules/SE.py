class SE(nn.Module):

    def __init__(self, channels: int , reduction: int = 16):

        super().__init__()
        assert channels >0
        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, kernel_size=1, bias=True)
        