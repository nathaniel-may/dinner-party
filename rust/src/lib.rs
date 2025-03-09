use argmin::{
    core::{CostFunction, Error, Executor, State},
    solver::simulatedannealing::{Anneal, SimulatedAnnealing},
};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
};
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
use wasm_bindgen::prelude::*;

#[derive(Clone)]
pub struct DinnerParty {
    pub participants: usize,
    // TODO problem could be expanded to include heterogeneous group sizes
    pub group_size: usize,
    pub rounds: usize,
    pub rng: Arc<Mutex<SmallRng>>,
}

impl DinnerParty {
    fn init(&self) -> Vec<Vec<Option<usize>>> {
        let mut seats: Vec<Option<usize>> = (0..self.participants).map(Some).collect();
        while seats.len() % self.group_size != 0 {
            seats.push(None)
        }

        // The first round is the ordered list of guests since all permutations are equally valuable
        let mut rounds = Vec::with_capacity(self.rounds);
        rounds.push(seats.clone());

        let mut rng = self.rng.lock().unwrap();
        while rounds.len() < self.rounds {
            seats.shuffle(&mut rng);
            rounds.push(seats.clone());
        }

        rounds
    }

    pub fn people_met(
        &self,
        assignment: &Vec<Vec<Option<usize>>>,
    ) -> HashMap<usize, HashSet<usize>> {
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

fn random_swap<A>(rng: &mut impl Rng, xs: &mut [A]) {
    if xs.len() <= 1 {
        return;
    }

    let i = rng.random_range(0..xs.len());
    let mut j = rng.random_range(0..xs.len());
    while i == j {
        j = rng.random_range(0..xs.len());
    }
    xs.swap(i, j);
}

#[test]
fn test_random_swap() {
    let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(1);
    let mut xs = vec![1, 2, 3, 4, 5];
    random_swap(&mut rng, &mut xs);
    assert_eq!(xs, vec![1, 2, 3, 5, 4])
}

pub fn run(
    problem: &DinnerParty,
    runs: u64,
) -> Result<Vec<Vec<Option<usize>>>, Box<dyn std::error::Error>> {
    let solver = SimulatedAnnealing::new(problem.participants as f64)?
        .with_stall_best(runs / 10)
        .with_reannealing_fixed(runs / 20);

    let res = Executor::new(problem.clone(), solver)
        .configure(|state| state.param(problem.init()).max_iters(runs))
        .run()
        .unwrap();

    let result = res.state().get_best_param().unwrap();

    Ok(result.clone())
}

#[wasm_bindgen()]
pub struct SeatAssignment {}

#[wasm_bindgen]
pub fn run_wasm(
    participants: usize,
    group_size: usize,
    rounds: usize,
    runs: u64,
) -> SeatAssignment {
    let problem = DinnerParty {
        participants,
        group_size,
        rounds,
        rng: Arc::new(Mutex::new(SeedableRng::from_os_rng())),
    };
    run(&problem, runs);
    SeatAssignment {}
}
