module Annealing where

import Prelude

import Data.Int (toNumber)
import Effect (Effect)
import Effect.Random (random) -- JS math.random()

type Parameters state =
    { max_steps :: Int
    -- higher numbers for more desirable states
    , energy :: state -> Number
    -- A random neighbor. The number is a random number between 0 and 1
    , neighbor :: state -> Number -> state
    , init_temperature :: Number
    -- Given the percentage of steps remaining (0,1], return the new temperature
    , annealing_schedule :: Number -> Number
    -- E(s) -> E(s_new) -> T -> Probability of acceptance
    , acceptance :: Number -> Number -> Number -> Number
    }

run :: ∀ s. Parameters s -> s -> Effect s
run = run' 0

run' :: ∀ s. Int -> Parameters s -> s -> Effect s
run' k config state =
    if k >= config.max_steps
    then pure state
    else do
        rand_x <- random
        rand_y <- random
        let t = config.annealing_schedule (toNumber $ k + 1 / config.max_steps)
            neighbor = config.neighbor state rand_x
        if rand_y <= config.acceptance (config.energy state) (config.energy neighbor) t
        then run' (k + 1) config neighbor
        else run' k config state
