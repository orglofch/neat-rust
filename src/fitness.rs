use std::collections::HashMap;

use genome::Population;

pub struct FitnessConfig {
    /// The function invoked to calculate fitness for members of the population.
    ///
    /// The function will be invoked with populations of size `fitness_batch_size`, partitioning
    /// the total population if necessary. It is assumed that `fitness_fn` invocations are
    /// independent and parallelizable.
    ///
    /// The function should return a mapping of the genome ids to their fitness.
    pub(crate) fitness_fn: fn(&Population) -> HashMap<u32, f32>,

    /// The size of the batches to partition the total population into when invoking the
    /// `fitness_fn`.
    fitness_batch_size: u32,
}

impl FitnessConfig {
    pub fn new(fitness_fn: fn(&Population) -> HashMap<u32, f32>) -> FitnessConfig {
        FitnessConfig {
            fitness_fn: fitness_fn,
            fitness_batch_size: 1,
        }
    }

    pub fn set_fitness_batch_size(&mut self, batch_size: u32) -> &mut FitnessConfig {
        self.fitness_batch_size = batch_size;
        self
    }
}
