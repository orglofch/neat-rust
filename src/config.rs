use std::collections::HashMap;

use checkpoint::CheckpointConfig;
use fitness::FitnessConfig;
use genome::{Genome, GenomeConfig, Population};
use speciation::SpeciationConfig;

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

pub struct Config {
    checkpoint_config: Option<CheckpointConfig>,
    fitness_config: FitnessConfig,
    genome_config: GenomeConfig,
    speciation_config: SpeciationConfig,

    population_size: u32,
}

impl Config {
    pub fn new(fitness_config: FitnessConfig, genome_config: GenomeConfig) -> Config {
        Config {
            checkpoint_config: None,
            fitness_config: fitness_config,
            genome_config: genome_config,
            speciation_config: SpeciationConfig::new(),
            population_size: 1,
        }
    }

    pub fn set_checkpoint_config(&mut self, config: CheckpointConfig) -> &mut Config {
        self.checkpoint_config = Some(config);
        self
    }

    pub fn set_speciation_config(&mut self, config: SpeciationConfig) -> &mut Config {
        self.speciation_config = config;
        self
    }

    pub fn set_population_size(&mut self, size: u32) -> &mut Config {
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

        for _ in 0..100000 {
            let fitness_by_id = (self.fitness_config.fitness_fn)(&population);

            // TODO(orglofch): We probavbly want a pool allocator for new genomes and connections.

            // TODO(orglofch): Possible change the way this is layed out depending on
            // the compatibility threshold.
            let mut species_by_proto_id: HashMap<u32, Vec<u32>> = HashMap::new();

            // Speciate the genomes into groups by some arbitrary protypical genomes.
            // TODO(orglofch): This whole iteration loop could be cleaned up.
            for (id, ref genome) in population.iter() {
                if species_by_proto_id.is_empty() {
                    species_by_proto_id.insert(*id, vec![*id]);
                } else {
                    // Find the first species that the genome is compatible with.
                    // If no such species can be found then create a new species.
                    // TODO(orglofch): Maybe find the best species?
                    let mut best_species = None;
                    for (proto_id, genomes) in species_by_proto_id.iter_mut() {
                        let proto_genome = population.get(&proto_id).unwrap();

                        let distance = proto_genome.distance(&genome, &self.speciation_config);

                        if distance <= self.speciation_config.compatibility_threshold {
                            best_species = Some(*id);
                            genomes.push(*id);
                            break;
                        }
                    }

                    // If a best species couldn't be found, create a new species for the genome
                    // using it as the prototypical genome.
                    species_by_proto_id.entry(best_species.unwrap_or(*id)).or_insert(Vec::new()).push(*id);
                }
            }

            // Calculate the weighted fitness of the population groups.
            let mut species_fitness_by_proto_id: HashMap<u32, f32> = HashMap::new();
            for (proto_id, genomes) in species_by_proto_id.iter() {
                let fitness_sum: f32 = genomes
                    .iter()
                // TODO(orglofch): Possible don't default to 0 or at least provide a warning that we're
                    // doing so.
                    .map(|id| fitness_by_id.get(&id).unwrap_or(&0.0))
                    .sum();
                species_fitness_by_proto_id.insert(*proto_id, fitness_sum / genomes.len() as f32);
            }

            // Perform intra-species crossover based on fitness.

            // Perform mutation.

            // Update the population pool.
            // TODO(orglofch): We probably want to try to reuse allocations here to speed it up.
        }
    }
}
