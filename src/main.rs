#[macro_use]
pub mod macros;

pub mod activation;
pub mod aggregation;
pub mod config;
pub mod checkpoint;
pub mod fitness;
pub mod gene;
pub mod genome;
pub mod innovation;
pub mod speciation;

use config::Config;
use fitness::FitnessConfig;
use genome::{GenomeConfig, Population};

use std::collections::HashMap;

const INPUT_1: &'static str = "input_1";
const INPUT_2: &'static str = "input_2";

const OUTPUT: &'static str = "output";

fn eval_fitness(population: &Population) -> HashMap<u32, f32> {
    let mut inputs: HashMap<&str, f32> = HashMap::new();
    inputs.insert(INPUT_1, 1.0);
    inputs.insert(INPUT_2, 1.0);

    return population
        .iter()
        .map(|(id, genome)| {
            (
                *id,
                *genome.activate(&inputs).get(&OUTPUT).unwrap(),
            )
        })
        .collect();
}

fn main() {
    let fitness_config = FitnessConfig::new(eval_fitness);

    let genome_config = GenomeConfig::new(vec![INPUT_1, INPUT_2], vec![OUTPUT]);

    let mut config = Config::new(fitness_config, genome_config);
    config.run();
}
