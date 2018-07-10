extern crate rand;

use std::collections::HashMap;

use checkpoint::CheckpointConfig;
use fitness::FitnessConfig;
use genome::{Genome, GenomeConfig, Population};
use speciation::{speciate, SpeciationConfig};

enum ConfigErrorType {
}

pub struct ConfigError {
    kind: ConfigErrorType,
}

impl ConfigError {
    pub fn description(&self) -> &str {
        match self.kind {
        }
    }
}

pub struct Config<'a> {
    checkpoint_config: Option<CheckpointConfig>,
    fitness_config: FitnessConfig,
    genome_config: GenomeConfig<'a>,
    speciation_config: SpeciationConfig,

    population_size: u32,
}

impl<'a> Config<'a> {
    pub fn new(fitness_config: FitnessConfig, genome_config: GenomeConfig<'a>) -> Config {
        Config {
            checkpoint_config: None,
            fitness_config: fitness_config,
            genome_config: genome_config,
            speciation_config: SpeciationConfig::new(),
            population_size: 1,
        }
    }

    pub fn set_checkpoint_config(&mut self, config: CheckpointConfig) -> &mut Config<'a> {
        self.checkpoint_config = Some(config);
        self
    }

    pub fn set_speciation_config(&mut self, config: SpeciationConfig) -> &mut Config<'a> {
        self.speciation_config = config;
        self
    }

    pub fn set_population_size(&mut self, size: u32) -> &mut Config<'a> {
        self.population_size = size;
        self
    }

    // TODO(orglofch): Split the runner out from the config.
    pub fn run(&mut self) {
        // Initialize population.
        let mut population: Population = HashMap::with_capacity(self.population_size as usize);

        for i in 0..self.population_size {
            population.insert(i as u32, Genome::new(&mut self.genome_config));
        }

        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            println!("ITER");

            let fitness_by_id = (self.fitness_config.fitness_fn)(&population);

            // TODO(orglofch): We probably want a pool allocator for new genomes and connections and
            // to reuse the existing allocated ganomes.

            let species = speciate(&population, &self.speciation_config);

            // Calculate the weighted fitness of the population groups to support fitness sharing.
            let species_fitness_by_proto_id: HashMap<u32, f32> = species
                .iter()
                .map(|(proto_id, genome_ids)| {
                    // TODO(orglofch): Consider not defaulting to 0, probably just assert.
                    let sum: f32 = genome_ids
                        .iter()
                        .map(|id| fitness_by_id.get(&id).unwrap_or(&0.0))
                        .sum();
                    (*proto_id, sum / genome_ids.len() as f32)
                })
                .collect();


            let fitness_sum: f32 = species_fitness_by_proto_id
                .iter()
                .map(|(_, fitness)| fitness)
                .sum();

            // Generate the new population.
            for (proto_id, species) in species.iter() {
                // This species generates offspring according to it shared normalized fitness.
                let new_species_size = population.len() as f32 * species_fitness_by_proto_id.get(&proto_id).unwrap() /
                    fitness_sum;

                // Perform crossover.
                //let children: Vec<&Genome> = species.iter()
                //    .map(|id| population.get_mut(&id))
                //    .collect();

                // Perform mutations.
                for id in species {
                    let genome = population.get_mut(&id).unwrap();

                    genome.mutate(&mut self.genome_config, &mut rng);
                }
            }
        }
    }
}
