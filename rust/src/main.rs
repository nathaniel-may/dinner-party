use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::simulatedannealing::{Anneal, SimulatedAnnealing};
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct DinnerParty {
    pub participants: usize,
    // TODO problem could be expanded to include heterogeneous group sizes
    pub group_size: usize,
    pub rounds: usize,
    // using SmallRng for speed
    pub rng: Arc<Mutex<SmallRng>>,
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

    pub fn min_people_met(&self, assignment: &Vec<Vec<Option<usize>>>) -> usize {
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

        let min = m.values().map(|x| x.len()).reduce(std::cmp::min);

        match min {
            None => 0,
            Some(x) => x,
        }
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

    // the number of people that the person who meets the least people is the cost.
    fn cost(&self, assignment: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.min_people_met(assignment) as f64)
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
        assert_eq!(2.0, cost)
    } else {
        panic!("cost function failed")
    }
}

pub fn approximate(problem: &mut DinnerParty, runs: u64) -> Vec<Vec<Option<usize>>> {
    let init = problem.init();
    let solver = SimulatedAnnealing::new(100.0).unwrap();
    let res = Executor::new(problem.clone(), solver)
        .configure(|state| {
            state
                .param(init)
                // stops after this number of iterations (runs in cli)
                .max_iters(runs)
        })
        // run the solver on the defined problem
        .run()
        .unwrap();

    res.state().get_best_param().unwrap().clone()
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
    let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(1);
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
    let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(1);
    let mut xs = vec![1, 2, 3, 4, 5];
    random_swap(&mut rng, &mut xs);
    assert_eq!(xs, vec![1, 2, 4, 3, 5])
}

fn main() {
    let runs = 1000000;
    let mut problem = DinnerParty {
        participants: 24,
        group_size: 6,
        rounds: 3,
        rng: Arc::new(Mutex::new(SeedableRng::from_os_rng())),
    };
    println!(
        "Running approximation for optimal Dinner Party of {} guests, tables of size {}, with {} rounds. Number of runs={runs}.",
        &problem.participants, &problem.group_size, &problem.rounds
    );
    let result = approximate(&mut problem, runs);
    println!(
        "Everyone meets at least {} people with the following assignments",
        problem.min_people_met(&result)
    );
    println!("{result:?}")
}
