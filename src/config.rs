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
    speciation_config: Option<SpeciationConfig>,

    population_size: u32,
}

impl Config {
    pub fn new(fitness_config: FitnessConfig, genome_config: GenomeConfig) -> Config {
        Config {
            checkpoint_config: None,
            fitness_config: fitness_config,
            genome_config: genome_config,
            speciation_config: None,
            population_size: 1,
        }
    }

    pub fn set_checkpoint_config(&mut self, config: CheckpointConfig) -> &mut Config {
        self.checkpoint_config = Some(config);
        self
    }

    pub fn set_speciation_config(&mut self, config: SpeciationConfig) -> &mut Config {
        self.speciation_config = Some(config);
        self
    }

    pub fn set_population_size(&mut self, size: u32) -> &mut Config {
        self.population_size = size;
        self
    }

    // TODO(orglofch): Split the runner out from the config.
    pub fn run(&mut self) {
        // Initialize population.
        let mut population: Population = Vec::with_capacity(self.population_size as usize);

        for i in 0..self.population_size {
            population.push(Genome::new(i, &mut self.genome_config));
        }

        for _ in 0..100000 {
            (self.fitness_config.fitness_fn)(&Vec::new());
        }
    }
}
