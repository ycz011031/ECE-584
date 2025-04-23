import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # TODO: Return the converted HardTanH
        assert isinstance(act_layer, nn.Hardtanh)
        return BoundHardTanh()

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        # TODO: Implement the linear lower and upper bounds for HardTanH you derived in Problem 4.2.
        """
         Hints: 
         1. Have a look at the section 3.2 of the CROWN paper [1] (Case Studies) as to how segments are made for multiple activation functions
         2. Look at the HardTanH graph, and see multiple places where the pre activation bounds could be located
         3. Refer the ReLu example in the class and the diagonals to compute the slopes(coefficients)/intercepts(bias)
         4. The paper talks about 3 segments S+, S- and S+- for sigmoid and tanh. You should figure your own segments based on preactivation bounds for hardtanh.
         [1] https://arxiv.org/pdf/1811.00866.pdf
        """

        batch_size, spec_size, hidden_size = last_uA.shape if last_uA is not None else last_lA.shape

        # Initialize new bounds
        uA = last_uA.clone() if last_uA is not None else None
        lA = last_lA.clone() if last_lA is not None else None
        ubias = torch.zeros((batch_size, spec_size), device=preact_lb.device)
        lbias = torch.zeros((batch_size, spec_size), device=preact_lb.device)

        # Create masks with correct shape (batch_size, spec_size, hidden_size)
        mask1 = (preact_ub <= -1).unsqueeze(1).expand(-1, spec_size, -1)
        mask2 = (preact_lb >= 1).unsqueeze(1).expand(-1, spec_size, -1)
        mask3 = ((preact_lb >= -1) & (preact_ub <= 1)).unsqueeze(1).expand(-1, spec_size, -1)
        mask4 = ((preact_lb < -1) & (preact_ub <= 1)).unsqueeze(1).expand(-1, spec_size, -1)
        mask5 = ((preact_lb >= -1) & (preact_lb < 1) & (preact_ub > 1)).unsqueeze(1).expand(-1, spec_size, -1)
        mask6 = ((preact_lb < -1) & (preact_ub > 1)).unsqueeze(1).expand(-1, spec_size, -1)

        # Case 1: u ≤ -1 → HardTanh(z) = -1 (constant)
        if mask1.any():
            if uA is not None:
                uA[mask1] = 0
                ubias[mask1.any(dim=2)] = -1
            if lA is not None:
                lA[mask1] = 0
                lbias[mask1.any(dim=2)] = -1

        # Case 2: l ≥ 1 → HardTanh(z) = 1 (constant)
        if mask2.any():
            if uA is not None:
                uA[mask2] = 0
                ubias[mask2.any(dim=2)] = 1
            if lA is not None:
                lA[mask2] = 0
                lbias[mask2.any(dim=2)] = 1

        # Case 4: l < -1 < u ≤ 1 → Lower bound is -1, upper bound follows identity
        if mask4.any():
            if lA is not None:
                lA[mask4] = 0
                lbias[mask4.any(dim=2)] = -1

        # Case 5: -1 ≤ l < 1 < u → Upper bound is 1, lower bound follows identity
        if mask5.any():
            if uA is not None:
                uA[mask5] = 0
                ubias[mask5.any(dim=2)] = 1

        # Case 6: l < -1 < u > 1 → Global relaxation using a linear bound
        if mask6.any():
            slope = (1 - (-1)) / (preact_ub - preact_lb)  # Compute slope
            bias = -slope * preact_lb - 1  # Compute bias

            # Expand dimensions to match (batch_size, spec_size, hidden_size)
            slope = slope.unsqueeze(1).expand(-1, spec_size, -1)
            bias = bias.unsqueeze(1).expand(-1, spec_size, -1)

            if uA is not None:
                uA[mask6] *= slope[mask6]
                ubias[mask6.any(dim=2)] = bias[mask6].sum(dim=-1)

            if lA is not None:
                lA[mask6] *= slope[mask6]
                lbias[mask6.any(dim=2)] = bias[mask6].sum(dim=-1)

        return uA, ubias, lA, lbias



