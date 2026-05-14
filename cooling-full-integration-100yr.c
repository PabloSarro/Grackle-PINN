/* ---------------------------------------------
 * Reproduce the Iliev06 Test 0 part 3 example:
 * Heat the gas with radiation for 0.5 Myr, then
 * let it cool for 5 more Myrs
 * --------------------------------------------- */

/* define these before including local headers like my_grackle_utils.h */
#define FIELD_SIZE 1

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grackle.h>


/* Hydrogen mass in g */
#define const_mh 1.67262171e-24
/* Electron mass in g */
#define const_me 9.10938e-28
/* erg/K */
#define const_kboltz 1.3806488e-16
/* erg * s */
#define const_planck_h 6.62606957e-27
/* cm / s */
#define const_speed_light_c 2.99792458e+10
/* year in s */
#define const_yr 31557600.
#define const_adiabatic_index 1.666667

#define SUCCESS 1
#define TINY_NUMBER 0




/*********************************************************************
/ Compute the mean molecular weight using the same rules as Grackle
*********************************************************************/

#define MU_METAL 16.0

int mean_weight_local_like_grackle(chemistry_data *my_chemistry,
                                   grackle_field_data *my_fields,
                                   gr_float *mu) {

  gr_float number_density = 0.;
  gr_float inv_metal_mol = 1.0 / MU_METAL;
  int i, dim, size = 1;

  for (dim = 0; dim < my_fields->grid_rank; dim++)
    size *= my_fields->grid_dimension[dim];

  if (!my_chemistry->use_grackle)
    return SUCCESS;

  for (i = 0; i < size; i++) {

    if (my_chemistry->primordial_chemistry > 0) {
      number_density =
          0.25 * (my_fields->HeI_density[i] + my_fields->HeII_density[i] +
                  my_fields->HeIII_density[i]) +
          my_fields->HI_density[i] + my_fields->HII_density[i] +
          my_fields->e_density[i];
    }

    /* Add in H2. */

    if (my_chemistry->primordial_chemistry > 1) {
      number_density +=
          my_fields->HM_density[i] +
          0.5 * (my_fields->H2I_density[i] + my_fields->H2II_density[i]);
    }

    if (my_fields->metal_density != NULL) {
      number_density += my_fields->metal_density[i] * inv_metal_mol;
    }

    /* Ignore deuterium. */

    mu[i] = my_fields->density[0] / number_density;
  }

  return SUCCESS;
}





/**
 * @brief Set up the units for grackle.
 **/
void setup_grackle_units(code_units *grackle_units_data, double density_units,
                         double length_units, double time_units) {

  grackle_units_data->comoving_coordinates = 0; /* no cosmo */
  grackle_units_data->density_units = density_units;
  grackle_units_data->length_units = length_units;
  grackle_units_data->time_units = time_units;
  grackle_units_data->a_units = 1.0;
  grackle_units_data->a_value = 1.;
  
  /* Set velocity units */
  set_velocity_units(grackle_units_data);

}

/**
 * @brief Set parameters for the grackle chemistry object
 **/
void setup_grackle_chemistry(chemistry_data *grackle_chemistry_data,
                             int primordial_chemistry, int UVbackground,
                             char *grackle_data_file, int use_radiative_cooling,
                             int use_radiative_transfer,
                             float hydrogen_fraction_by_mass) {

  /* Set parameter values for chemistry. */
  grackle_chemistry_data->use_grackle = 1; /* chemistry on */
  grackle_chemistry_data->with_radiative_cooling = use_radiative_cooling;
  grackle_chemistry_data->primordial_chemistry = primordial_chemistry;
  grackle_chemistry_data->dust_chemistry = 0;
  grackle_chemistry_data->metal_cooling = 0;
  grackle_chemistry_data->UVbackground = UVbackground;
  grackle_chemistry_data->grackle_data_file = grackle_data_file;
  grackle_chemistry_data->use_radiative_transfer = use_radiative_transfer;
  grackle_chemistry_data->HydrogenFractionByMass = hydrogen_fraction_by_mass;
  grackle_chemistry_data->Gamma =
      const_adiabatic_index; /* defined in const.h */
      
  grackle_chemistry_data->CaseBRecombination = 1;
    
}

/**
 * @brief Set up grackle gas fields.
 **/
void setup_grackle_fields(grackle_field_data *grackle_fields,
                          gr_float species_densities[12],
                          gr_float interaction_rates[5], double gas_density,
                          double internal_energy) {

  int *dimension = malloc(3 * sizeof(int));
  int *start = malloc(3 * sizeof(int));
  int *end = malloc(3 * sizeof(int));

  /* Set grid dimension and size.
   * grid_start and grid_end are used to ignore ghost zones. */
  grackle_fields->grid_rank = 3;
  grackle_fields->grid_dimension = dimension;
  grackle_fields->grid_start = start;
  grackle_fields->grid_end = end;
  /* used only for H2 self-shielding approximation */
  /* grackle_fields->grid_dx = 0.0; */

  /* NOTE: if you're trying to simplify this, you MUST allocate GRIDDIM = 3
   * and grid_dimension, grid_start, grid_end with at least 3D as well.
   * Otherwise, grackle will cause segfaults because they do pointer arithmetics
   * assuming 3 dimensions internally. */
  for (int i = 0; i < 3; i++) {
    /* the active dimension not including ghost zones. */
    grackle_fields->grid_dimension[i] = 1;
    grackle_fields->grid_start[i] = 0;
    grackle_fields->grid_end[i] = 0;
  }
  grackle_fields->grid_dimension[0] = FIELD_SIZE;
  grackle_fields->grid_end[0] = FIELD_SIZE - 1;

  /* Set initial quantities */
  grackle_fields->density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->internal_energy = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->x_velocity = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->y_velocity = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->z_velocity = malloc(FIELD_SIZE * sizeof(gr_float));
  /* for primordial_chemistry >= 1 */
  grackle_fields->HI_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->HII_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->HeI_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->HeII_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->HeIII_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->e_density = malloc(FIELD_SIZE * sizeof(gr_float));
  /* for primordial_chemistry >= 2 */
  grackle_fields->HM_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->H2I_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->H2II_density = malloc(FIELD_SIZE * sizeof(gr_float));
  /* for primordial_chemistry >= 3 */
  grackle_fields->DI_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->DII_density = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->HDI_density = malloc(FIELD_SIZE * sizeof(gr_float));
  /* for metal_cooling = 1 */
  grackle_fields->metal_density = malloc(FIELD_SIZE * sizeof(gr_float));

  /* volumetric heating rate (provide in units [erg s^-1 cm^-3]) */
  grackle_fields->volumetric_heating_rate =
      malloc(FIELD_SIZE * sizeof(gr_float));
  /* specific heating rate (provide in units [egs s^-1 g^-1] */
  grackle_fields->specific_heating_rate = malloc(FIELD_SIZE * sizeof(gr_float));

  /* radiative transfer ionization / dissociation rate fields (provide in units
   * [1/s]) */
  grackle_fields->RT_HI_ionization_rate = malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->RT_HeI_ionization_rate =
      malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->RT_HeII_ionization_rate =
      malloc(FIELD_SIZE * sizeof(gr_float));
  grackle_fields->RT_H2_dissociation_rate =
      malloc(FIELD_SIZE * sizeof(gr_float));
  /* radiative transfer heating rate field (provide in units [erg s^-1 cm^-3])
   */
  grackle_fields->RT_heating_rate = malloc(FIELD_SIZE * sizeof(gr_float));

  for (int i = 0; i < FIELD_SIZE; i++) {

    /* initial density */
    grackle_fields->density[i] = gas_density;

    /* initial internal energy using initial temperature */
    grackle_fields->internal_energy[i] = internal_energy;
    /* T / (grackle_chemistry_data.Gamma - 1.0) / mean_weight /
     * temperature_units; */

    grackle_fields->HI_density[i] = species_densities[0];
    grackle_fields->HII_density[i] = species_densities[1];
    grackle_fields->HeI_density[i] = species_densities[2];
    grackle_fields->HeII_density[i] = species_densities[3];
    grackle_fields->HeIII_density[i] = species_densities[4];
    grackle_fields->e_density[i] =
        species_densities[5]; /* electron density*mh/me */

    grackle_fields->HM_density[i] = species_densities[6];
    grackle_fields->H2I_density[i] = species_densities[7];
    grackle_fields->H2II_density[i] = species_densities[8];
    grackle_fields->DI_density[i] = species_densities[9];
    grackle_fields->DII_density[i] = species_densities[10];
    grackle_fields->HDI_density[i] = species_densities[11];

    /* solar metallicity */
    grackle_fields->metal_density[i] = 0.0;
    /* grackle_chemistry_data.SolarMetalFractionByMass *
     * grackle_fields->density[i]; */

    grackle_fields->x_velocity[i] = 0.0;
    grackle_fields->y_velocity[i] = 0.0;
    grackle_fields->z_velocity[i] = 0.0;

    grackle_fields->volumetric_heating_rate[i] = 0.0;
    grackle_fields->specific_heating_rate[i] = 0.0;

    grackle_fields->RT_heating_rate[i] = interaction_rates[0];
    grackle_fields->RT_HI_ionization_rate[i] = interaction_rates[1];
    grackle_fields->RT_HeI_ionization_rate[i] = interaction_rates[2];
    grackle_fields->RT_HeII_ionization_rate[i] = interaction_rates[3];
    grackle_fields->RT_H2_dissociation_rate[i] = interaction_rates[4];
  }
}



/**
 * @brief Deallocate fields when you're done.
 **/
void clean_up_fields(grackle_field_data *grackle_fields) {

  free(grackle_fields->grid_dimension);
  free(grackle_fields->grid_start);
  free(grackle_fields->grid_end);

  free(grackle_fields->density);
  free(grackle_fields->internal_energy);
  free(grackle_fields->x_velocity);
  free(grackle_fields->y_velocity);
  free(grackle_fields->z_velocity);
  free(grackle_fields->HI_density);
  free(grackle_fields->HII_density);
  free(grackle_fields->HeI_density);
  free(grackle_fields->HeII_density);
  free(grackle_fields->HeIII_density);
  free(grackle_fields->e_density);
  free(grackle_fields->HM_density);
  free(grackle_fields->H2I_density);
  free(grackle_fields->H2II_density);
  free(grackle_fields->DI_density);
  free(grackle_fields->DII_density);
  free(grackle_fields->HDI_density);
  free(grackle_fields->metal_density);
  free(grackle_fields->volumetric_heating_rate);
  free(grackle_fields->specific_heating_rate);

  free(grackle_fields->RT_HI_ionization_rate);
  free(grackle_fields->RT_HeI_ionization_rate);
  free(grackle_fields->RT_HeII_ionization_rate);
  free(grackle_fields->RT_H2_dissociation_rate);
  free(grackle_fields->RT_heating_rate);
}






/**
 * @brief Dump the setup used to generate the example
 *
 * @param fp FILE pointer to write into.
 * @param grackle_fields grackle field data
 * @param grackle_chemistry_data grackle chemistry data.
 * @param mass_units mass units that convert internal units to cgs
 * @param length_units length units that convert internal units to cgs
 * @param velocity_units velocity units that convert internal units to cgs
 * @param dt current time step
 * @param hydrogen_fraction_by_mass hydrogen fraction by mass used.
 * @param gas_density gas density used. In internal units.
 * @param internal_energy internal energy used. In internal units.
 **/
void write_my_setup(FILE *fd, grackle_field_data grackle_fields,
                    chemistry_data grackle_chemistry_data, double mass_units,
                    double length_units, double velocity_units, double dt,
                    double hydrogen_fraction_by_mass, double gas_density,
                    double internal_energy) {
  fprintf(fd, "# Result file created using grackle standalone program.\n");
  fprintf(fd, "# mass units used: %.6g [g]\n", mass_units);
  fprintf(fd, "# length units used: %.6g [cm]\n", length_units);
  fprintf(fd, "# velocity units units used: %.6g [cm/s]\n", velocity_units);
  fprintf(fd, "# dt used: %.6g [internal units]\n", dt);
  fprintf(fd, "# hydrogen mass fraction used: %.6g\n",
          hydrogen_fraction_by_mass);
  fprintf(fd, "# gas density used: %.6g [internal units]\n", gas_density);
  fprintf(fd, "# inital internal energy used: %.6g [internal units]\n",
          internal_energy);
  fprintf(fd, "# Grackle parameters:\n");
  fprintf(fd, "# grackle_chemistry_data.use_grackle = %d\n",
          grackle_chemistry_data.use_grackle);
  fprintf(fd, "# grackle_chemistry_data.with_radiative_cooling %d\n",
          grackle_chemistry_data.with_radiative_cooling);
  fprintf(fd, "# grackle_chemistry_data.primordial_chemistry = %d\n",
          grackle_chemistry_data.primordial_chemistry);
  fprintf(fd, "# grackle_chemistry_data.dust_chemistry = %d\n",
          grackle_chemistry_data.dust_chemistry);
  fprintf(fd, "# grackle_chemistry_data.metal_cooling = %d\n",
          grackle_chemistry_data.metal_cooling);
  fprintf(fd, "# grackle_chemistry_data.UVbackground = %d\n",
          grackle_chemistry_data.UVbackground);
  fprintf(fd, "# grackle_chemistry_data.CaseBRecombination = %d\n",
          grackle_chemistry_data.CaseBRecombination);
  fprintf(fd, "# grackle_chemistry_data.grackle_data_file = %s\n",
          grackle_chemistry_data.grackle_data_file);
  fprintf(fd, "# grackle_chemistry_data.use_radiative_transfer = %d\n",
          grackle_chemistry_data.use_radiative_transfer);
  fprintf(fd, "# grackle_chemistry_data.use_volumetric_heating_rate = %d\n",
          grackle_chemistry_data.use_volumetric_heating_rate);
  fprintf(fd, "# grackle_chemistry_data.use_specific_heating_rate = %d\n",
          grackle_chemistry_data.self_shielding_method);
  fprintf(fd, "# grackle_chemistry_data.self_shielding_method = %d\n",
          grackle_chemistry_data.use_specific_heating_rate);          
  fprintf(fd, "# grackle_chemistry_data.HydrogenFractionByMass = %.3g\n",
          grackle_chemistry_data.HydrogenFractionByMass);
  fprintf(fd, "# grackle_chemistry_data.Gamma = %.6g\n",
          grackle_chemistry_data.Gamma);
  fprintf(fd, "# Grackle field data:\n");

#define write_grackle_field(v)                                                 \
  if (grackle_fields.v != NULL)                                                \
  fprintf(fd, "# grackle_fields." #v " = %g\n", grackle_fields.v[0])

  write_grackle_field(density);
  write_grackle_field(internal_energy);
  write_grackle_field(HI_density);
  write_grackle_field(HII_density);
  write_grackle_field(HeI_density);
  write_grackle_field(HeII_density);
  write_grackle_field(HeIII_density);
  write_grackle_field(e_density);
  write_grackle_field(HM_density);
  write_grackle_field(H2I_density);
  write_grackle_field(H2II_density);
  write_grackle_field(DI_density);
  write_grackle_field(DII_density);
  write_grackle_field(HDI_density);
  write_grackle_field(metal_density);
  write_grackle_field(x_velocity);
  write_grackle_field(y_velocity);
  write_grackle_field(z_velocity);
  write_grackle_field(volumetric_heating_rate);
  write_grackle_field(specific_heating_rate);
  write_grackle_field(RT_HI_ionization_rate);
  write_grackle_field(RT_HeI_ionization_rate);
  write_grackle_field(RT_HeII_ionization_rate);
  write_grackle_field(RT_H2_dissociation_rate);
  write_grackle_field(RT_heating_rate);
}




/**
 * @brief write header to a file/stdout
 **/
void write_header(FILE *fd) {

  fprintf(fd,
          "#%8s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n",
          "step", "Time [yr]", "dt [yr]", "Temperature [K]", "Mean M Wgt [1]",
          "Tot dens [IU]", "HI dens [IU]", "HII dens [IU]", "HeI dens [IU]",
          "HeII dens [IU]", "HeIII dens [IU]", "e- n. dens [IU]",
          "IntrnEnerg [IU]");
}

/**
 * @brief write the current state of a field with index i to a file/stdout
 **/
void write_timestep(FILE *fd, grackle_field_data *grackle_fields,
                    code_units *grackle_units_data,
                    chemistry_data *grackle_chemistry_data, int field_index,
                    double t, double dt, double time_units, int step) {

  /* Additional arrays to store temperature and mean molecular weights
   * of each cell. */
  gr_float temperature[FIELD_SIZE];
  gr_float mu[FIELD_SIZE];
  for (int i = 0; i < FIELD_SIZE; i++) {
    temperature[i] = 0.;
    mu[i] = 0.;
  }

  /* Grab temperature and mean molecular weights. */
  if (calculate_temperature(grackle_units_data, grackle_fields, temperature) ==
      0) {
    fprintf(stderr, "Error in calculate_temperature.\n");
    abort();
  }

  if (mean_weight_local_like_grackle(grackle_chemistry_data, grackle_fields,
                                     mu) != SUCCESS) {
    fprintf(stderr, "Error in local_calculate_mean_weight.\n");
    abort();
  }

  fprintf(fd,
          "%9d %15.3e %15.3e %15.3e %15.3e %15.3e %15.3e %15.3e %15.3e %15.3e "
          "%15.3e "
          "%15.3e %15.3e\n",
          step, t / const_yr * time_units, dt / const_yr * time_units,
          temperature[field_index], mu[field_index],
          grackle_fields->density[field_index],
          grackle_fields->HI_density[field_index],
          grackle_fields->HII_density[field_index],
          grackle_fields->HeI_density[field_index],
          grackle_fields->HeII_density[field_index],
          grackle_fields->HeIII_density[field_index],
          grackle_fields->e_density[field_index],
          grackle_fields->internal_energy[field_index]);
}






/******************************************************************
 * 
 *  M A I N
 * 
 *****************************************************************/




int main(int argc, char *argv[]) {
  if (argc < 5) {
      printf("Usage: %s <T0> <n0> <xHII> <out_name>\n", argv[0]);
      return 1;
  }

  double initial_T = atof(argv[1]);
  double initial_density = atof(argv[2]);
  double initial_xHII = atof(argv[3]);
  char *output_filename = argv[4];


  /******************************************************************
   * Set up initial conditions and runtime parameters.
   *****************************************************************/

  /* Print some extra data to screen? */
  int verbose = 1;
  grackle_verbose = 0;

  /* output file */
  FILE *fd = fopen(output_filename, "w"); // output_fi_test_stable.dat

  /* Define units  */
  /* --------------*/
  double mass_units = 1.98841e43;       /* 10^10 M_sun in grams */
  double length_units = 3.08567758e24;  /* Mpc in centimeters */
  double velocity_units = 1.e5;         /* km/s in centimeters per second */

  double density_units = mass_units / (length_units * length_units * length_units);
  double time_units = length_units / velocity_units;
  
  /* Define timesteps to write to output file */
  double dt_sample = 100.0 * const_yr / time_units;
  double t_next_sample = dt_sample;
  

  /* Time integration variables */
  /* -------------------------- */

  double dt_max_cool = 1e2; /* upper limit for time during cooling, in yr.    normaly 1e2 */
  double dt_max_heat = 10.; /* upper limit for time during heating, in yr.    normally 10. */
  double dt_init = 9.549e+04;   /* max dt while heating. in yr. Will be converted later */
  double tinit = 0;     /* in yr; will be converted later */
  double tend = 2.5e6;      /* in yr; will be converted later */
  /* Convert times to internal units. */
  double t = tinit * const_yr / time_units;  /* yr to code units */
  tend = tend * const_yr / time_units;       /* yr to code units */
  dt_init = dt_init * const_yr / time_units; /* yr to code units */
  dt_max_cool = dt_max_cool * const_yr / time_units;
  dt_max_heat = dt_max_heat * const_yr / time_units;

  /* Set up initial conditions for gas cells */
  /* --------------------------------------- */
  double hydrogen_fraction_by_mass = 1;
  
  /* Iliev test values */
  //double gas_density = const_mh; /* in cgs, will be converted later. Corresponds to number density 1 cm^-3 */
  //double T = 100.;               /* K */

  //double gas_density = const_mh; /* in cgs, will be converted later. Corresponds to number density 1 cm^-3 */
  double T = initial_T; // 1.601e+05;               /* K */

  


  /* Derived quantities from ICs */
  /* --------------------------- */

  /* Assuming fully neutral hydrogen gas */ 
  //double mu_init = 1.;
  // double HI_density_init = 1.292e-02*1.67262171e-24; //9.816e-03*1.67262171e-24;
  // double HII_density_init = 3.092e-06*1.67262171e-24; //2.350e-06*1.67262171e-24;
  double HI_fraction = 1.0 - initial_xHII;// HI_density_init / (HI_density_init + HII_density_init);
  double HII_fraction = initial_xHII; // HII_density_init / (HI_density_init + HII_density_init);
  
  double mu_init = 1.0 / (1.0 + HII_fraction);
  double internal_energy_cgs = T * const_kboltz / (const_mh * mu_init * (const_adiabatic_index - 1.));
  double internal_energy = internal_energy_cgs / (velocity_units * velocity_units);
  
  double HI_density_init = HI_fraction * initial_density * const_mh;
  double HII_density_init = HII_fraction * initial_density * const_mh;

  double gas_density = HI_density_init+HII_density_init;
  gas_density /= density_units;


  /* define the hydrogen number density */
  /* use `gr_float` to use the same type of floats that grackle
   * is compiled in. Compile grackle with precision-32 if you want
   * floats, or precision-64 otherwise. */
  gr_float nH = hydrogen_fraction_by_mass * gas_density / (const_mh / mass_units);
  gr_float nHI, nHII, nHeI, nHeII, nHeIII, ne;

  /* get densities of primordial spicies assuming ionization equilibrium */
  //ionization_equilibrium_calculate_densities(T, nH, hydrogen_fraction_by_mass,
  //                                           &nHI, &nHII, &nHeI, &nHeII,
  //                                           &nHeIII, &ne);
  
  nHI   = nH * HI_fraction;
  nHII  = nH * HII_fraction;
  //nHI = nH;
  //nHII = TINY_NUMBER;
  nHeI  = TINY_NUMBER;
  nHeII = TINY_NUMBER;
  nHeIII= TINY_NUMBER;
  ne    = nH * HII_fraction;
  //ne = TINY_NUMBER;
  
  
  
  

  gr_float HI_density = nHI * (const_mh / mass_units);
  gr_float HII_density = nHII * (const_mh / mass_units);
  gr_float HeI_density = nHeI * (4 * const_mh / mass_units);
  gr_float HeII_density = nHeII * (4 * const_mh / mass_units);
  gr_float HeIII_density = nHeIII * (4 * const_mh / mass_units);
  /* !! this is the convention adopted by Grackle
   * !! e_density is the electron number density multiplied by proton mass,
   * !! or electron mass density * nH / ne */
  gr_float e_density = ne * (const_mh / mass_units);

  /* Store them all in a single array for simplicity. */
  gr_float species_densities[12] = {HI_density,
                                    HII_density,
                                    HeI_density,
                                    HeII_density,
                                    HeIII_density,
                                    e_density,
                                    TINY_NUMBER * gas_density,
                                    TINY_NUMBER * gas_density,
                                    TINY_NUMBER * gas_density,
                                    TINY_NUMBER * gas_density,
                                    TINY_NUMBER * gas_density,
                                    TINY_NUMBER * gas_density};





  /* Grackle behaviour setup */
  /* ----------------------- */

  int UVbackground = 0;         /* toogle on/off the UV background */
  int primordial_chemistry = 1; /* choose the chemical network */
  int use_radiative_cooling = 1;
  int use_radiative_transfer = 1;
  char *grackle_data_file = "";

  /*********************************************************************
   * Set up gracke data and fields.
   **********************************************************************/

  /* Units  */
  /* ------ */

  /* First, set up the units system. We assume cgs
   * These are conversions from code units to cgs. */
  code_units grackle_units_data;
  setup_grackle_units(&grackle_units_data, density_units, length_units,
                      time_units);

  /* Chemistry Parameters */
  /* -------------------- */

  chemistry_data grackle_chemistry_data;
  if (set_default_chemistry_parameters(&grackle_chemistry_data) == 0) {
    fprintf(stderr, "Error in set_default_chemistry_parameters");
    return EXIT_FAILURE;
  }

  /* Set parameter values for chemistry. */
  setup_grackle_chemistry(&grackle_chemistry_data, primordial_chemistry,
                          UVbackground, grackle_data_file,
                          use_radiative_cooling, use_radiative_transfer,
                          hydrogen_fraction_by_mass);


  if (initialize_chemistry_data(&grackle_units_data) == 0) {
    fprintf(stderr, "Error in initialize_chemistry_data.\n");
    return EXIT_FAILURE;
  }

  /* Gas Data */
  /* -------- */

  gr_float interaction_rates[5] = {0., 0., 0., 0., 0};

  /* Create struct for storing grackle field data */
  grackle_field_data grackle_fields;
  setup_grackle_fields(&grackle_fields, species_densities, interaction_rates,
                       gas_density, internal_energy);

  /* Write headers */
  /* ------------- */

  if (verbose) {
    printf("%15s%15s%15s%15s%15s%15s%15s%15s\n",
           "Initial setup: ", "Temperature", "nHI", "nHII", "nHeI", "nHeII",
           "nHeIII", "ne");
    printf("%15s%15g%15g%15g%15g%15g%15g%15g\n\n", "Initial setup: ", T, nHI,
           nHII, nHeI, nHeII, nHeIII, ne);
  }

  write_header(stdout);
  //write_timestep(stdout, &grackle_fields, &grackle_units_data,
  //               &grackle_chemistry_data, /*field_index=*/0, t, dt_max_heat,
  //               time_units, /*step=*/0);

  /* write down what ICs you used into file */
  write_my_setup(fd, grackle_fields, grackle_chemistry_data, mass_units,
                 length_units, velocity_units, dt_max_heat,
                 hydrogen_fraction_by_mass, gas_density, internal_energy);
  
  write_header(fd);
  write_timestep(fd, &grackle_fields, &grackle_units_data,
                 &grackle_chemistry_data, 0, t, dt_max_heat,
                 time_units, /*step=*/0);

  /*********************************************************************
  / Calling the chemistry solver
  / These routines can now be called during the simulation.
  *********************************************************************/

  int step = 0;
  double dt = dt_init;
  double dt_use;
  double tol = 1e-7*const_yr/time_units;
  while (t < tend) {

    /* Set up radiation fields, and compute the resulting interaction
     * rates depending on the simulation time. */

    gr_float iact_rates[5] = {0., 0., 0., 0., 0.};

    dt_use = fmin(dt, dt_max_cool);

    if (t / const_yr * time_units < 0.5e6) {
    //if (t / const_yr * time_units < 0.5e10) {
      /* below 0.5 Myr, we heat. */

      //iact_rates[0] = 1.65117e-17;                /* heating rate          */
      //iact_rates[1] = 5.03133e+13;                /* HI-ionization rate    */    

      iact_rates[0] = 0;                /* heating rate          */
      iact_rates[1] = 0;                /* HI-ionization rate    */   

      dt_use = fmin(dt, dt_max_heat);
    }

    /* Increase timestep size */
    dt *= 1.001;

    for (int i = 0; i < FIELD_SIZE; i++) {
      grackle_fields.RT_heating_rate[i]         = iact_rates[0];
      grackle_fields.RT_HI_ionization_rate[i]   = iact_rates[1];
      grackle_fields.RT_HeI_ionization_rate[i]  = iact_rates[2];
      grackle_fields.RT_HeII_ionization_rate[i] = iact_rates[3];
      grackle_fields.RT_H2_dissociation_rate[i] = iact_rates[4];
    }

    /* Get cooling time */
    gr_float tchem_time;
    if (local_calculate_cooling_time(&grackle_chemistry_data, &grackle_rates,
                                     &grackle_units_data, &grackle_fields,
                                     &tchem_time) == 0) {

      fprintf(stderr, "Error in calculate_cooling_time.");
      abort();
    }
    dt_use = fmin(0.1 * fabs(tchem_time), dt_use);

    t += dt_use;
    step += 1;

    if (local_solve_chemistry(&grackle_chemistry_data, &grackle_rates,
                              &grackle_units_data, &grackle_fields,
                              dt_use) == 0) {
      fprintf(stderr, "Error in solve_chemistry.\n");
      return EXIT_FAILURE;
    }

    //write_timestep(stdout, &grackle_fields, &grackle_units_data,
    //               &grackle_chemistry_data, /*field_index=*/0, t, dt_use,
    //               time_units, step);

    if (t+tol >= t_next_sample) { // Output to file every dt=100 yr.
      write_timestep(fd, &grackle_fields, &grackle_units_data,
                     &grackle_chemistry_data, 0, t, dt_sample,
                     time_units, step);

      t_next_sample += dt_sample;
    }
  }

  /* Cleanup */
  fclose(fd);
  clean_up_fields(&grackle_fields);
  free_chemistry_data();



  return EXIT_SUCCESS;
}
