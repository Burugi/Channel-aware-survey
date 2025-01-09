from layers import adv_loss

class TransferLoss(object):
    def __init__(self, loss_type='cosine', input_dim=512):
        """
        Supported loss_type: mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv
        """
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):
        """Compute adaptation loss

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix

        Returns:
            [tensor] -- transfer loss
        """
        if self.loss_type == 'adv':
            loss = adv_loss.adv(X, Y, input_dim=self.input_dim, hidden_dim=32)

        return loss

if __name__ == "__main__":
    import torch
    trans_loss = TransferLoss('adv')
    a = (torch.randn(5,512) * 10).cuda()
    b = (torch.randn(5,512) * 10).cuda()
    print(trans_loss.compute(a, b))
