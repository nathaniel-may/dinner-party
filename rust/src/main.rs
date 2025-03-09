use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::simulatedannealing::{Anneal, SimulatedAnnealing};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct DinnerParty {
    pub participants: usize,
    // TODO problem could be expanded to include heterogeneous group sizes
    pub group_size: usize,
    pub rounds: usize,
    // using SmallRng for speed
    pub rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl DinnerParty {
    pub fn init(&self) -> Vec<Vec<Option<usize>>> {
        let mut seats: Vec<Option<usize>> = (0..self.participants).map(Some).collect();
        while seats.len() % self.group_size != 0 {
            seats.push(None)
        }

        // The first round is the ordered list of guests since all permutations are equally valuable
        let mut rounds = Vec::with_capacity(self.rounds);
        rounds.push(seats.clone());

        let mut rng = self.rng.lock().unwrap();
        while rounds.len() < self.rounds {
            random_permutation(&mut rng, &mut seats);
            rounds.push(seats.clone());
        }

        rounds
    }

    fn people_met(&self, assignment: &Vec<Vec<Option<usize>>>) -> HashMap<usize, HashSet<usize>> {
        let mut m: HashMap<usize, HashSet<usize>> = HashMap::with_capacity(self.participants);
        for round in assignment {
            for i in 0..self.participants / self.group_size {
                let table: Vec<usize> = round
                    [i * self.group_size..i * self.group_size + self.group_size]
                    .iter()
                    .flat_map(|x| x.iter())
                    .cloned()
                    .collect();

                for person in &table {
                    match m.get_mut(person) {
                        Some(set) => {
                            for p in &table {
                                set.insert(*p);
                            }
                        }
                        None => {
                            m.insert(*person, HashSet::<usize>::from_iter(table.clone()));
                        }
                    }
                }
            }
        }
        m
    }

    pub fn min_people_met(&self, people_met: &HashMap<usize, HashSet<usize>>) -> usize {
        people_met
            .values()
            .map(|x| x.len())
            .reduce(std::cmp::min)
            .unwrap_or(0)
    }

    // converts a seating chart into ordered table names for each person
    pub fn to_table_order(
        &self,
        assignment: &Vec<Vec<Option<usize>>>,
    ) -> HashMap<usize, Vec<char>> {
        let mut m: HashMap<usize, Vec<char>> = HashMap::new();

        for round in assignment {
            let mut table_names: VecDeque<char> = ('A'..='Z').collect();
            for table in round.chunks(self.group_size) {
                let table_name = table_names.pop_front().unwrap();
                let people: Vec<usize> = table.iter().flat_map(|x| x.iter()).cloned().collect();

                for person in &people {
                    match m.get_mut(person) {
                        Some(tables) => {
                            tables.push(table_name);
                        }
                        None => {
                            m.insert(*person, vec![table_name]);
                        }
                    }
                }
            }
        }

        m
    }

    pub fn print_csv(
        &self,
        assignment: &Vec<Vec<Option<usize>>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtr = csv::Writer::from_writer(std::io::stdout());

        let mut ordered_output: Vec<(usize, Vec<char>)> =
            self.to_table_order(assignment).drain().collect();
        ordered_output.sort();

        for row in ordered_output.clone() {
            let person: usize = row.0;
            let tables: Vec<String> = row.1.iter().map(ToString::to_string).collect();
            let row = [&[person.to_string()], &tables[..]].concat();
            wtr.write_record(row)?;
        }

        wtr.flush()?;
        Ok(())
    }
}

impl Anneal for DinnerParty {
    type Param = Vec<Vec<Option<usize>>>;

    type Output = Vec<Vec<Option<usize>>>;

    type Float = f64;

    fn anneal(&self, param: &Self::Param, temp: Self::Float) -> Result<Self::Output, Error> {
        let mut rng = self.rng.lock().unwrap();
        let mut next_assignment = Vec::with_capacity(param.len());
        for round in param {
            let mut round_stepped = round.clone();
            for _ in 0..(temp as usize) % self.participants {
                random_swap(&mut rng, &mut round_stepped)
            }
            next_assignment.push(round_stepped);
        }
        Ok(next_assignment)
    }
}

impl CostFunction for DinnerParty {
    type Param = Vec<Vec<Option<usize>>>;
    type Output = f64;

    // minimizing the number of unmet people for the person who has met the least people
    fn cost(&self, assignment: &Self::Param) -> Result<Self::Output, Error> {
        let people_met = self.people_met(assignment);
        let min_people_met = self.min_people_met(&people_met);
        let mean_people_met =
            people_met.values().map(|x| x.len()).sum::<usize>() as f64 / people_met.len() as f64;
        // favor solutions with higher averages if they ahve the same minimum people met
        let adjustment = (mean_people_met - min_people_met as f64) / 100.0;
        Ok((self.participants - min_people_met) as f64 - adjustment)
    }
}

#[test]
fn test_cost_fn() {
    let party = DinnerParty {
        participants: 6,
        group_size: 2,
        rounds: 2,
        rng: Arc::new(Mutex::new(rand::SeedableRng::seed_from_u64(1))),
    };
    let assignment = vec![
        vec![0, 1, 2, 3, 4, 5].into_iter().map(Some).collect(),
        vec![0, 1, 2, 4, 3, 5].into_iter().map(Some).collect(),
    ];
    if let Ok(cost) = party.cost(&assignment) {
        assert_eq!(3.993333333333333, cost)
    } else {
        panic!("cost function failed")
    }
}

fn random_permutation<A>(rng: &mut impl RngCore, xs: &mut [A]) {
    if xs.len() <= 1 {
        return;
    }
    for i in 0..(xs.len() - 1) {
        let swap_i = rng.next_u32() as usize % (xs.len() - 1 - i);
        xs.swap(i, swap_i);
    }
}

#[test]
fn test_random_permutation() {
    let mut rng: Xoshiro256PlusPlus = rand::SeedableRng::seed_from_u64(1);
    let mut xs = vec![1, 2, 3, 4, 5];
    random_permutation(&mut rng, &mut xs);
    assert_eq!(xs, vec![1, 3, 4, 2, 5])
}

fn random_swap<A>(rng: &mut impl RngCore, xs: &mut [A]) {
    if xs.len() <= 1 {
        return;
    }
    let i = rng.next_u32() as usize % (xs.len() - 1);
    let mut j = rng.next_u32() as usize % (xs.len() - 1);
    while i == j {
        j = rng.next_u32() as usize % (xs.len() - 1);
    }
    xs.swap(i, j);
}

#[test]
fn test_random_swap() {
    let mut rng: Xoshiro256PlusPlus = rand::SeedableRng::seed_from_u64(1);
    let mut xs = vec![1, 2, 3, 4, 5];
    random_swap(&mut rng, &mut xs);
    assert_eq!(xs, vec![1, 2, 4, 3, 5])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runs = 1000000;
    let problem = DinnerParty {
        participants: 18,
        group_size: 6,
        rounds: 3,
        rng: Arc::new(Mutex::new(SeedableRng::from_os_rng())),
    };
    println!(
        "Running approximation for optimal Dinner Party of {} guests, tables of size {}, with {} rounds. Number of runs={runs}.",
        &problem.participants, &problem.group_size, &problem.rounds
    );

    let solver = SimulatedAnnealing::new(problem.participants as f64)?
        .with_stall_best(runs / 10)
        .with_reannealing_fixed(runs / 20);
    let res = Executor::new(problem.clone(), solver)
        .configure(|state| state.param(problem.init()).max_iters(runs))
        .run()
        .unwrap();

    let result = res.state().get_best_param().unwrap().clone();

    println!(
        "Everyone meets at least {} people with the following assignments",
        problem.min_people_met(&problem.people_met(&result))
    );
    problem.print_csv(&result)
}
