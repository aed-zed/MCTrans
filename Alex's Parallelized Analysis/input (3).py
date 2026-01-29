#!/usr/bin/env python3

from scipy.constants import e, c

from pywarpx import picmi, warpx, particle_containers, libwarpx
from pywarpx.callbacks import installafterInitEsolve, installafterstep

import numpy
import argparse
import sys
import cupy
from mpi4py import MPI as mpi
import os
from scipy.constants import m_u, physical_constants

comm = mpi.COMM_WORLD

################################INPUT PARAMETERS################################
authors = "Alex Gargone <agargone@avalanche.energy"

m_e = picmi.constants.m_e
m_p = picmi.constants.m_p
clight = picmi.constants.c
q_e = picmi.constants.q_e
kb = physical_constants['Boltzmann constant in eV/K'][0]

########GRID SIZING########
grid_scaling_factor = 1
Lx = 11.2e-2 # Diameter of the simulation box
nx = 112*grid_scaling_factor # Number of cells in x
Ly = 10.4e-2 # Diameter of the simulation box
ny = 104*grid_scaling_factor # Number of cells in x
Lz = 0.8e-2
nz = 8*grid_scaling_factor
dx = Lx/nx
dz = Lz/nz
###########################

##########TIMING###########
# dt = 2.0e-11/grid_scaling_factor
# dt = 2.0e-12/grid_scaling_factor
# dt = 1.923907368e-12/grid_scaling_factor
dt = 5.0e-12/grid_scaling_factor
# last_time_step = int(float(1000.0e-9)/dt)+10
last_time_step = int(float(20.0e-6)/dt)+10
flux_end = -1
###########################

####ORBITRON PARAMETERS####
cathode_voltage = -100.00e3 #V
b_field_mag = 0.646 #T
cathode_radius = (-Ly/2)
anode_radius = Ly/2
E_field_mag = 961538.4615
###########################

####PARTICLE PARAMETERS####
particle_density_cm3 = 1e18 #/1cm3
particle_density_m3 = particle_density_cm3*1e6 #/1m3
num_macroparticles = 1.0e6
Amps = 0.0
initial_gas_temp = 300 #K
initial_gas_energy_eV = kb*initial_gas_temp
ion_current = Amps*6.241509074460762e18
electron_current =  Amps*6.241509074460762e18
Dplus_mass = (2.014102*m_u) - (m_e)
Dplus_u_gaussian_PICMI = numpy.sqrt(2*q_e*initial_gas_energy_eV/(Dplus_mass*clight*clight)) * clight
elec_u_gaussian_PICMI = numpy.sqrt(2*q_e*initial_gas_energy_eV/(m_e*clight*clight)) * clight
###########################

#################################SIM PARAMETERS#################################

#########SIM SIZING########
grid = picmi.Cartesian3DGrid(
    number_of_cells = [nx, ny, nz],
    warpx_blocking_factor = 8,
    warpx_max_grid_size = nx,
    lower_boundary_conditions = ['neumann', 'periodic', 'periodic'],
    upper_boundary_conditions = ['neumann', 'periodic', 'periodic'],
    lower_boundary_conditions_particles = ['absorbing', 'periodic', 'periodic'],
    upper_boundary_conditions_particles = ['absorbing', 'periodic', 'periodic'],
    lower_bound = [-Lx/2, -Ly/2, -Lz/2],
    upper_bound = [ Lx/2,  Ly/2,  Lz/2]
)
###########################

##########E-FIELD##########
# solver = picmi.ElectromagneticSolver(
#    grid = grid,
#    method = 'Yee'
# )
###########################
solver = picmi.ElectrostaticSolver(
    grid = grid, 
    method = 'Multigrid', 
    required_precision = 1.0e-4,
    maximum_iterations = 1000,
    # warpx_magnetostatic = True,
    warpx_absolute_tolerance = 3.0e-1
    # warpx_absolute_tolerance = 3.0e-5
)
 
###########################
boundary_expression = f"0 + 2*(x<={cathode_radius}) + 2*(x>={anode_radius}) - (x>{cathode_radius})*(x<{anode_radius})"
###########################    
potential_expression = f"{cathode_voltage}*(x<=({cathode_radius}+2.5e-3))"
###########################

############STL############
embedded_boundary = picmi.EmbeddedBoundary(
    # potential = potential_expression,
    implicit_function = boundary_expression,
    cover_multiple_cuts = True,
)
###########################

##########E-FIELD##########
Efield_slope = (-E_field_mag)/(anode_radius-cathode_radius)
Efield = picmi.AnalyticInitialField(
    Ex_expression = f"0 + (-({Efield_slope}*x)-{E_field_mag})*(x>={cathode_radius})*(x<={anode_radius})",
    Ey_expression = 0,
    Ez_expression = 0
)
###########################

##########B-FIELD##########
Bfield = picmi.ConstantAppliedField(
    Bz = b_field_mag
)
###########################

#######FIELD SMOOTHER######
#Binomial_field_smoother = picmi.BinomialSmoother(
#   n_pass  = [2, 2, 2]
#)
###########################

##############################PARTICLE PARAMETERS###############################
# n_mp_per_cell = 2
n_mp_per_cell = 80
Plasma_rmax = anode_radius-2.5e-3
Plasma_rmin = anode_radius-3.5e-3
SA = (Plasma_rmax-Plasma_rmin)*Lz
Volume = ((Plasma_rmax-Plasma_rmin)**2)*Lz
num_cells = (Volume/(dx*dx*dz))
num_macroparticles = (2*n_mp_per_cell)*num_cells
num_particles = (2*particle_density_m3)*Volume
macro_weight = (num_particles)/num_macroparticles
num_particles_check = num_macroparticles*macro_weight
print(macro_weight)


DT_density_expression = f"(({particle_density_m3})*(x<={Plasma_rmax})*(x>={Plasma_rmin})) + (0*(x>{Plasma_rmax})) + (0*(x<{Plasma_rmin}))"
elec_density_expression = f"(({particle_density_m3})*(x<={Plasma_rmax})*(x>={Plasma_rmin})) + (0*(x>{Plasma_rmax})) + (0*(x<{Plasma_rmin}))"
#############################################################
DTplus_Layout = picmi.PseudoRandomLayout(
    n_macroparticles_per_cell = n_mp_per_cell,
    grid = grid
)
elecs_Layout = picmi.PseudoRandomLayout(
    n_macroparticles_per_cell = n_mp_per_cell,
    grid = grid
)
#############################################################
Dplus_initial_distribution = picmi.AnalyticDistribution(
    lower_bound = [Plasma_rmin, -Ly/2, -Lz/2],
    upper_bound = [Plasma_rmax,  Ly/2,  Lz/2],
    density_expression = DT_density_expression,
    warpx_density_min = 1e2,
    rms_velocity = [Dplus_u_gaussian_PICMI, Dplus_u_gaussian_PICMI, Dplus_u_gaussian_PICMI],
)

elec_initial_distribution = picmi.AnalyticDistribution(
    lower_bound = [Plasma_rmin, -Ly/2, -Lz/2],
    upper_bound = [Plasma_rmax,  Ly/2,  Lz/2],
    density_expression = elec_density_expression,
    warpx_density_min = 1e2,
    rms_velocity = [elec_u_gaussian_PICMI, elec_u_gaussian_PICMI, elec_u_gaussian_PICMI],
)

#############################################################
Dplus = picmi.Species(
    warpx_save_particles_at_eb  = 1,
    warpx_save_particles_at_xlo = 1,
    warpx_save_particles_at_xhi = 1,
    warpx_save_particles_at_ylo = 1,
    warpx_save_particles_at_yhi = 1,
    warpx_save_particles_at_zlo = 1,
    warpx_save_particles_at_zhi = 1,
    method = "Boris",
    name = "Dplus",
    particle_type = 'D',
    charge_state = 1,
    warpx_do_not_deposit = True,
    initial_distribution = [Dplus_initial_distribution],
    warpx_add_int_attributes = {
        'Start_it':f"(t/{dt})"
        },
    warpx_add_real_attributes = {
        # 'Start_radius':"(((x*x)+(y*y))**0.5)", 
        'Start_x':"(x)", 
        'Start_y':"(y)", 
        'Start_z':"(z)", 
        'Start_ux':"(ux)", 
        'Start_uy':"(uy)", 
        'Start_uz':"(uz)",
        'Start_t':"(t)" 
        }
)
#############################################################
electrons = picmi.Species(
    warpx_save_particles_at_eb  = 1,
    warpx_save_particles_at_xlo = 1,
    warpx_save_particles_at_xhi = 1,
    warpx_save_particles_at_ylo = 1,
    warpx_save_particles_at_yhi = 1,
    warpx_save_particles_at_zlo = 1,
    warpx_save_particles_at_zhi = 1,
    method = "Boris",
    name = "electrons",
    particle_type = 'electron',
    warpx_do_not_deposit = True,
    initial_distribution = [elec_initial_distribution],
    warpx_add_int_attributes = {
        'Start_it':f"(t/{dt})"
        },
    warpx_add_real_attributes = {
        # 'Start_radius':"(((x*x)+(y*y))**0.5)", 
        'Start_x':"(x)", 
        'Start_y':"(y)", 
        'Start_z':"(z)", 
        'Start_ux':"(ux)", 
        'Start_uy':"(uy)", 
        'Start_uz':"(uz)",
        'Start_t':"(t)" 
        }
)
#############################################################
#############################################################

##############################COLLISION PARAMETERS##############################
###########################
Coulomb_ee = picmi.CoulombCollisions(
    name='Coulomb_ee',
    species=[electrons, electrons],
    ndt = 1
)
###########################
Coulomb_DD = picmi.CoulombCollisions(
    name='Coulomb_DD',
    species=[Dplus, Dplus],
    ndt = 1
)
###########################
Coulomb_De = picmi.CoulombCollisions(
    name='Coulomb_De',
    species=[Dplus, electrons],
    ndt = 1
)
###########################


#############################DIAGNOSTICS PARAMETERS#############################
Particle_Density = picmi.ParticleFieldDiagnostic(
    name = "Particle_Density",
    func = f"1/({dx}*{dx}*{dz})",
    do_average = False
)
PpC = picmi.ParticleFieldDiagnostic(
    name = "Particles_per_Cell",
    func = f"1/({macro_weight})",
    do_average = False
)
###########################
diag_Field = picmi.FieldDiagnostic(
    name = 'diag',
    grid = grid,
    period = "1:5000:500, 0::1.0e4",
    warpx_format = 'openpmd',
    warpx_openpmd_backend = 'h5',
    warpx_file_min_digits = 10,
    data_list = ['Ex', 'Ey', 'Ez', 'rho_electrons', 'rho_Dplus', 'T_electrons', 'T_Dplus'],
    # data_list = ['Ex', 'Ey', 'Ez', 'rho_electrons', 'rho_Dplus', 'T_electrons', 'T_Dplus', 'phi'],
    # data_list = ['Ex', 'Ey', 'Ez', 'rho_electrons', 'rho_Dplus', 'rho_Tplus', 'T_electrons', 'T_Dplus', 'T_Tplus', 'part_per_cell'],
    # data_list = ['Ex', 'Ey', 'Ez', 'rho_electrons', 'rho_Dplus', 'rho_Tplus', 'Jx', 'Jy', 'Jz', 'T_electrons', 'T_Dplus', 'T_Tplus', 'phi'],
    # data_list = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'rho_electrons', 'rho_Dplus', 'rho_Tplus', 'Jx', 'Jy', 'Jz', 'T_electrons', 'T_Dplus', 'T_Tplus', 'phi'],
    warpx_particle_fields_to_plot = [Particle_Density, PpC]
)
###########################
diag_Particle = picmi.ParticleDiagnostic(
    name = 'diag',
    period = "1:5000:500, 0::1.0e4",
    # period = "1:5000:100, 0::5.0e3, 4.5e4:5.0e4:100, 2.5e4:3.0e4:100",
    warpx_format = 'openpmd',
    warpx_openpmd_backend = 'h5',
    warpx_file_min_digits = 10
)
###########################

###########################
diag_boundary = picmi.ParticleBoundaryScrapingDiagnostic(
    name = 'diag_boundary',
    period = 5e2,
    warpx_format = 'openpmd',
    warpx_openpmd_backend = 'h5',
    warpx_file_min_digits = 10
)
###########################
checkpoints = picmi.Checkpoint(
    name = 'checkpoints',
    period = 1.0e5,
#    step_min = 0,
    warpx_file_min_digits = 10
)
###########################
PE = picmi.ReducedDiagnostic(
    diag_type = 'ParticleEnergy',
    name = 'PE',
    period = "0:20:1, 0::100"
)
###########################
PN = picmi.ReducedDiagnostic(
    diag_type = 'ParticleNumber',
    name = 'PN',
    period = "0:20:1, 0::100"
)
###########################
diag_Particle_trace = picmi.ParticleDiagnostic(
    name = 'diag_trace',
    period = "1::100",
    # period = "1:20000:100",
    warpx_format = 'openpmd',
    warpx_openpmd_backend = 'h5',
    warpx_file_min_digits = 10,
    # warpx_random_fraction = 0.5
)
###########################


#################################INITIALIZE SIM#################################
sim = picmi.Simulation(
    solver = solver,
    max_steps = int(last_time_step),
    warpx_embedded_boundary = embedded_boundary,
    warpx_collisions = [Coulomb_ee, Coulomb_DD, Coulomb_De],
    warpx_field_gathering_algo=  'momentum-conserving',
    particle_shape = 'cubic',
    warpx_use_filter = False,
    warpx_break_signals = 'HUP',
    warpx_used_inputs_file = 'warpx_used_inputs.txt',
    time_step_size = dt,
    warpx_numprocs = [2, 2, 1],
    # warpx_amr_restart = "./diags/checkpoints0000320497"
)

sim.add_applied_field(
    Efield
)

sim.add_applied_field(
    Bfield
)

sim.add_species(
    Dplus,
    layout = [DTplus_Layout]
)

sim.add_species(
    electrons,
    layout = [elecs_Layout]
)

sim.add_diagnostic(diag_Field)
sim.add_diagnostic(diag_Particle)

# sim.add_diagnostic(diag_Particle_trace)

sim.add_diagnostic(diag_boundary)

sim.add_diagnostic(checkpoints)

sim.add_diagnostic(PE)
sim.add_diagnostic(PN)


sim.write_input_file(
    file_name  = './input_file'
)

######################################################################################################################################################################

class ParticleNumberCorrector(object):
    """
    Object that changes the electrostatic field at each timestep, so
    as to ensure that we maintain the required potential difference between
    the anode and cathode.
    """
    
    def __init__(self, sim, xmin, xmax, ymin, ymax, zmin, zmax):
        self.sim = sim
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.species_info = {}
        pass

    def register_initial_particle_count(self):
        print('hi')
        # this_proc_check=False
        for sp in self.sim.species:
            name = sp.name
            pcw = particle_containers.ParticleContainerWrapper(name)
            count = int(pcw.get_particle_count(local=False))
            weights = pcw.get_particle_weight()
            
            has_local = 0
            w_local = 0.0
            if len(weights)>0 and len(weights[0])>0:
                w_local = float(weights[0][0])
                has_local = 1
                            
            w_sum   = comm.allreduce(w_local, op=mpi.SUM)
            has_sum = comm.allreduce(has_local, op=mpi.SUM)
            weight = (w_sum / has_sum) if has_sum > 0.0 else 1.0
            
            m = sp.mass
            if m == "m_e":
                m = m_e
            Temp = float(numpy.sqrt(2*q_e*initial_gas_energy_eV/(m*clight*clight)) * clight)          
            self.species_info[name] = {"target": count, "weight": weight, "Temp": Temp}
            print(f"[Init] {name} initial count = {count}, with weight: {weight}, with t: {Temp}")

            
    def reinject_species(self):
        for sp in self.sim.species:
            species_name = sp.name
            pcw = particle_containers.ParticleContainerWrapper(species_name)
            target = self.species_info[species_name]["target"]
            current = pcw.get_particle_count(local=False)
            missing = float(target - current)
            if missing > 0:
                x = numpy.random.uniform(self.xmin, self.xmax, size=int(missing))
                y = numpy.random.uniform(self.ymin, self.ymax, size=int(missing))
                z = numpy.random.uniform(self.zmin, self.zmax, size=int(missing))
                
                ux = numpy.random.normal(loc=0, scale=float(self.species_info[species_name]["Temp"]), size=int(missing))
                uy = numpy.random.normal(loc=0, scale=float(self.species_info[species_name]["Temp"]), size=int(missing))
                uz = numpy.random.normal(loc=0, scale=float(self.species_info[species_name]["Temp"]), size=int(missing))

                particle_attributes = {'Start_x':(x), 'Start_y':(y), 'Start_z':(z), 'Start_ux':(ux/clight), 'Start_uy':(uy/clight), 'Start_uz':(uz/clight)}

                pcw.add_particles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=(self.species_info[species_name]["weight"]*numpy.ones(int(missing))), unique_particles=False, **particle_attributes)

                print(f"[Reinject] At it: {self.sim.extension.warpx.getistep(0)}, Rank #{libwarpx.amr.ParallelDescriptor.MyProc()} Added {missing} particles to {species_name} in annulus, total was {current} now is {pcw.get_particle_count(local=False)}, should be {target}")



pnc = ParticleNumberCorrector(sim, Plasma_rmax, (Plasma_rmax+1e-3), (-Ly/2), (Ly/2), (-Lz/2), (Lz/2))
installafterInitEsolve( pnc.register_initial_particle_count )
# installafterstep( pnc.reinject_species )
######################################################################################################################################################################


sim.step()