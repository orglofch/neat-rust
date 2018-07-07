use std::collections::{HashMap, HashSet};

use genome::{Genome, GenomeConfig, Population};

pub struct SpeciationConfig {
    /// The maximum compatible distance between two genomes which are still part of the same species.
    pub compatibility_threshold: f32,

    // TODO(orglofch): Maybe add compatibility_excess_coefficient: f32.
    /// The coefficient applied to the number of disjoint genes when calculating compatibility.
    pub compatibility_disjoint_coefficient: f32,

    /// The coefficient applied to the weight difference of matching genes when calculating compatibility.
    pub compatibility_weight_coefficient: f32,
}

// TODO(orglofch): Create appropriate defaults.
impl SpeciationConfig {
    pub fn new() -> SpeciationConfig {
        SpeciationConfig {
            compatibility_threshold: 0.0,
            compatibility_disjoint_coefficient: 0.0,
            compatibility_weight_coefficient: 0.0,
        }
    }

    pub fn set_compatibility_threshold(&mut self, threshold: f32) -> &mut SpeciationConfig {
        self.compatibility_threshold = threshold;
        self
    }

    pub fn set_compatibility_disjoint_coefficient(&mut self, coefficient: f32) -> &mut SpeciationConfig {
        self.compatibility_disjoint_coefficient = coefficient;
        self
    }

    pub fn set_compatibility_weight_coefficient(&mut self, coefficient: f32) -> &mut SpeciationConfig {
        self.compatibility_weight_coefficient = coefficient;
        self
    }
}

/// Speciate a population into a set of sub-species.
///
/// All `Genome` instances in a species are measured against some arbitrary "prototypical `Genome` s.t.
/// all of the `Genome` instances in a species are within a certain "distance" of the proto-genome.
///
/// # Arguments
///
/// * `population` - The population of `Genome` instances to speciate.
///
/// * `speciation_config` - The config to use in speciating the `Genome` instances.
pub fn speciate(population: &Population, speciation_config: &SpeciationConfig) -> HashMap<u32, HashSet<u32>> {
    // TODO(orglofch): Guess number of species if this is sufficiently large.
    let mut species_by_proto_id: HashMap<u32, HashSet<u32>> = HashMap::new();

    // Speciate the genomes into groups by some arbitrary protypical genomes.
    // TODO(orglofch): This whole iteration loop could be cleaned up.
    for (id, ref genome) in population.iter() {
        let mut best_species = *id;
        if !species_by_proto_id.is_empty() {
            // Find the first species that the genome is compatible with.
            // If no such species can be found then create a new species.
            // TODO(orglofch): Maybe find the best species?
            for proto_id in species_by_proto_id.keys() {
                let proto_genome = population.get(&proto_id).unwrap();

                let distance = proto_genome.distance(&genome, speciation_config);

                if distance <= speciation_config.compatibility_threshold {
                    best_species = *proto_id;
                    break;
                }
            }
        };

        // If a best species couldn't be found, create a new species for the genome
        // using it as the prototypical genome.
        species_by_proto_id
            .entry(best_species)
            .or_insert(HashSet::new())
            .insert(*id);
    }

    return species_by_proto_id;
}

#[cfg(test)]
mod test {
    use super::*;

    const INPUT: &'static str = "input";
    const OUTPUT: &'static str = "output";

    #[test]
    fn test_speciate() {
        let mut gen_conf = GenomeConfig::new(vec![INPUT], vec![OUTPUT]);
        gen_conf.set_start_connected(false);

        let gen_1 = Genome::new(&mut gen_conf);

        gen_conf.set_start_connected(true);

        let gen_2 = Genome::new(&mut gen_conf);

        let gen_3 = Genome::new(&mut gen_conf);

        let mut population: Population = HashMap::new();
        population.insert(1, gen_1);
        population.insert(2, gen_2);
        population.insert(3, gen_3);

        let mut speciation_conf = SpeciationConfig::new();
        speciation_conf.set_compatibility_disjoint_coefficient(1.0);
        speciation_conf.set_compatibility_threshold(0.0); // Genomes must be identical.

        let species = speciate(&population, &speciation_conf);

        assert_eq!(species.len(), 2);
        assert!(
            species
                .values()
                .find(|val| **val == vec![1].into_iter().collect())
                .is_some()
        );
        assert!(
            species
                .values()
                .find(|val| **val == vec![2, 3].into_iter().collect())
                .is_some()
        );
    }
}
