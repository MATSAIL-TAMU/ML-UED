import xtal
import numpy as np
import sys
import random
import argparse
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import copy
import glob
from multiprocessing import Pool
from functools import partial

# Parse input arguments
parser = argparse.ArgumentParser(description='Random UED Image Generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--phonon-path', help='Path to phonon struture folder')
parser.add_argument('--ucref', default='', help='Path to reference unit cell')
parser.add_argument('--num', type=int, default=10, help='Number of images to generate')
parser.add_argument('--save-folder', help='Path for saving images')
parser.add_argument('--suffix', help='Suffix for images')
parser.add_argument('--add-defects', action='store_true', help='Flag to add vacancy defects to structures')
parser.add_argument('--add-degradation', action='store_true', help='Flag to add degradation to structures')
parser.add_argument('--add-distortion', action='store_true', help='Flag to add unit cell distortion to structures')
parser.add_argument('--add-rotation', action='store_true', help='Flag to add unit cell rotation to structures')

args = parser.parse_args()
phonon_structure_paths = glob.glob(args.phonon_path + '/PHON_CONTCAR_*')
phonon_structure_paths.sort()

def gen_structure(unit_cell_path, particle_size, vac_fraction, disp_delta, shear_strain, rotation_angle):

    orig = xtal.AtTraj()
    orig.read_snapshot_vasp(unit_cell_path)
    orig.make_dircar_matrices()

    # PERIODIC
    #---------
    periodic_x, periodic_y = particle_size
    orig.make_periodic([periodic_x, periodic_y, 1])
    orig.make_dircar_matrices()
    orig.dirtocar()

    # DEFECTS
    #---------
    non_vac_fraction = 100 - vac_fraction
    atoms_mo = [atom for atom in orig.snaplist[0].atomlist if atom.element=='MO']
    atoms_te = [atom for atom in orig.snaplist[0].atomlist if atom.element=='TE']
    atoms = random.sample(atoms_mo, int(len(atoms_mo)*non_vac_fraction/100.0)) + random.sample(atoms_te, int(len(atoms_te)*non_vac_fraction/100.0))
    orig.snaplist[0].atomlist = atoms

    # THERMAL
    #---------
    for atom in orig.snaplist[0].atomlist:
        displace_vector = (np.random.rand(3) - 0.5) * disp_delta
        atom.cart += displace_vector
    orig.cartodir()

    # DISTORT
    #---------
    shear_x, shear_y = shear_strain
    scale_factor = np.array([shear_x, shear_y, 1.0])
    orig.box = np.multiply(orig.box, scale_factor)

    # ROTATE
    #--------
    all_atom_positions = [[atom.cart[0], atom.cart[1]] for atom in orig.snaplist[0].atomlist]
    rotation_center = np.mean(np.array(all_atom_positions), axis=0)
    orig.rotate(rotation_center, rotation_angle)

    return orig






# Plot UED pattern
def make_ued_pattern(snapshot, intensity_level, snapshot2 = None, interpolate = 0.0):

    gx_range = np.arange(-1.3,1.305,0.01)
    gy_range = np.arange(-1.3,1.305,0.01)
    GX, GY = np.meshgrid(gx_range, gy_range)

    # Create atomic form factors
    q_sqr = ((4*np.pi*np.pi*np.power(GX,2)) + (4*np.pi*np.pi*np.power(GY,2)))/(4.0*4.0*np.pi*np.pi)
    aff_Mo = (3.7025*np.exp(-0.2772*q_sqr)) + (17.2356*np.exp(-1.0958*q_sqr)) + (12.8876*np.exp(-11.004*q_sqr)) + (3.7429*np.exp(-61.6584*q_sqr)) + 4.3875
    aff_Te = (19.9644*np.exp(-4.81742*q_sqr)) + (19.0138*np.exp(-0.420885*q_sqr)) + (6.14487*np.exp(-28.5284*q_sqr)) + (2.5239*np.exp(-70.8403*q_sqr)) + 4.352
    aff = {'MO':aff_Mo, 'TE':aff_Te}

    # Create Baseline image
    mask = ((np.power(GX,2) + np.power(GY,2)) > 0.0225)
    F_g1 = np.zeros_like(GX) + (np.zeros_like(GX) * 1j)
    F_g2 = np.zeros_like(GX) + (np.zeros_like(GX) * 1j)

    for atom in snapshot.atomlist:
        exponent = ((GX * atom.cart[0])+(GY * atom.cart[1])) * 2.0 * np.pi * (-1.0j)
        F_g1 += aff[atom.element] * np.exp(exponent) * np.exp(0.0 - q_sqr*0.5*np.pi*np.pi)

    if snapshot2 is not None:
        for atom in snapshot2.atomlist:
            exponent = ((GX * atom.cart[0])+(GY * atom.cart[1])) * 2.0 * np.pi * (-1.0j)
            F_g2 += aff[atom.element] * np.exp(exponent) * np.exp(0.0 - q_sqr*0.5*np.pi*np.pi)

    F_g = ((1.0-interpolate)*F_g1) + (interpolate*F_g2)

    UED = np.square(np.absolute(F_g))
    noise_amplitude_filter = gaussian_filter(UED, sigma=6)
    UED = gaussian_filter(UED, sigma=4)
    UED = np.multiply(UED, mask)

    noise_matrix = np.random.normal(0.5, 0.5, UED.shape)
    UED = UED + np.multiply(np.multiply(noise_matrix, noise_amplitude_filter), mask)

    UED = UED * intensity_level

    return UED




def loopable(structure_id, UEDavg = None):
    structure_path = phonon_structure_paths[structure_id]

    particle_size = np.rint(np.random.normal(1,0.2,2) * np.array([9,5])).astype(int)
    thermal_displacement = np.random.uniform(low = 0.0, high = 0.3)

    if args.add_defects:
        vac_fraction = np.random.randint(0,20)
    else:
        vac_fraction = 0

    if args.add_degradation:
        vac_fraction = np.random.randint(75,99)
    else:
        vac_fraction = 0

    if args.add_distortion:
        shear_strain = np.random.normal(1,0.09,2)
    else:
        shear_strain = np.array([0.0, 0.0])

    if args.add_rotation:
        rotation_angle = np.random.uniform(low = 0.0-(np.pi/3.0), high = np.pi/3.0)
    else:
        rotation_angle = 0.0

    structure = gen_structure(structure_path, particle_size, vac_fraction, thermal_displacement, shear_strain, rotation_angle)

    intensity_level = np.random.gamma(2,1)
    intensity_level = np.maximum(intensity_level, 0.1)  # Set intensity level to at least 0.1
    UED = make_ued_pattern(structure.snaplist[0], intensity_level)

    filename = 'UED_' + args.suffix + '_%5.5d' % structure_id + '.png'
    os.makedirs(args.save_folder, exist_ok = True)
    filepath = os.path.join(args.save_folder, filename)

    sizes = np.shape(UED)


    if args.ucref != '':
        UEDdiff = UED - UEDavg
        UEDpos = copy.copy(UEDdiff)
        UEDneg = copy.copy(UEDdiff)
        UEDzero = copy.copy(UEDdiff)

        UEDpos[UEDpos < 0] = 0
        UEDpos = UEDpos/np.max(UEDpos)

        UEDneg[UEDneg > 0] = 0
        UEDneg = UEDneg/np.min(UEDneg)

        UEDzero = UEDzero * 0
        UEDdiff = np.array([UEDpos, UEDzero, UEDneg]).T

        fig = plt.figure(figsize=(1,1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        # UEDdiff = np.transpose(UEDdiff, (1, 2, 0))
        ax.imshow(UEDdiff)#, cmap='gray')
        plt.savefig(filepath, dpi=sizes[0], transparent=False)

    else:

        fig = plt.figure(figsize=(1,1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(UED, cmap='gray')
        plt.savefig(filepath, dpi=sizes[0], transparent=False)



#if args.ucref != '':
def gen_ref_UED(id_number):
    particle_size = np.rint(np.random.normal(1,0.2,2) * np.array([51,30])).astype(int)
    thermal_displacement = np.random.uniform(low = 0.0, high = 0.3)
    structure = gen_structure(args.ucref, particle_size, 0, thermal_displacement, np.array([0.0, 0.0]), 0.0)
    intensity_level = np.random.gamma(2,1)
    intensity_level = np.maximum(intensity_level, 0.1)  # Set intensity level to at least 0.1
    UED = make_ued_pattern(structure.snaplist[0], intensity_level)
    return UED



if __name__ == "__main__":
    mypool = Pool(4)

    if args.ucref != '':
        g = mypool.map(gen_ref_UED, range(3))
        g = np.array(g)
        print(g.shape)
        UEDavg = np.mean(g, axis=0)
        print(UEDavg.shape)

        partial_loop = partial(loopable, UEDavg = UEDavg)
        #mypool.map(partial_loop, range(len(phonon_structure_paths)))
        mypool.map(partial_loop, range(args.num))
    else:
        #mypool.map(loopable, range(len(phonon_structure_paths)))
        mypool.map(loopable, range(args.num))
