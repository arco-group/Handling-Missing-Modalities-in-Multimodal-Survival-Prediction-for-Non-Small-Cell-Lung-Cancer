from pytorch_tabnet.tab_network import TabNet

__all__ = ["TabNetTorch"]


class TabNetTorch(TabNet):

    def __init__(self, *args, **kwargs):
        super(TabNetTorch, self).__init__(*args, **kwargs)
        self.input_size = self.input_dim
        self.output_size = self.output_dim

    def forward(self, x, *_):
        return super().forward(x)[0]


if __name__ == "__main__":
    pass
