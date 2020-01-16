"""Test the symmetry between the gradient and divergence operators."""

import pathlib
import unittest
import yaml

import pyibm


class GradientDivergenceTestCase(unittest.TestCase):
    """Tests for the gradient and divergence operators."""

    def setUp(self):
        """Setup."""
        # Load grid configuration from file.
        rootdir = pathlib.Path(__file__).absolute().parent
        datadir = rootdir / 'data'
        filepath = datadir / 'mesh.yaml'
        with open(filepath, 'r') as infile:
            config = yaml.load(infile, Loader=yaml.FullLoader)['mesh']
        # Create staggered grids.
        grid = pyibm.GridBase(config=config)  # vertices
        gridc = pyibm.GridCellCentered(grid=grid)  # cell-centered grid
        gridx = pyibm.GridFaceX(grid=grid)  # x-face centered grid
        gridy = pyibm.GridFaceY(grid=grid)  # y-face centered grid
        # Assemble gradient and divergence operators.
        GHat = pyibm.assemble_GHat(gridc, gridx, gridy)
        DHat = pyibm.assemble_DHat(gridc, gridx, gridy)
        # Assemble scaling diagonal operators.
        MHat = pyibm.assemble_MHat(gridx, gridy)
        R = pyibm.assemble_R(gridx, gridy)
        RInv = pyibm.diagonal_inv(R)
        # Normalize gradient and divergence operators.
        self.G = MHat @ GHat
        self.D = DHat @ RInv

    def test_nnz(self):
        """Test value of non-zero elements."""
        # Check same number of non-zeros.
        self.assertEqual(self.G.nnz, self.D.nnz)
        # Check nnz in {-1, 1}.
        G_abs = abs(self.G)
        self.assertEqual(G_abs.data.min(), G_abs.data.min())
        D_abs = abs(self.D)
        self.assertEqual(D_abs.data.min(), D_abs.data.min())

    def test_symmetry(self):
        """Test symmetry between gradient and divergence operators."""
        # Check D = - G.T.
        K = self.D + self.G.T
        K = K.multiply(abs(K) > 1e-12)  # discard very small values
        self.assertTrue(K.nnz == 0)
