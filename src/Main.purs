module Main where

import Prelude

import Control.Monad.Error.Class (throwError)
import Data.Foldable (traverse_)
import Data.Int.Bits (xor)
import Data.Maybe (maybe)
import Data.Time.Duration (Milliseconds(..))
import Effect (Effect)
import Effect.Aff (Aff, delay, launchAff_)
import Effect.Aff.Class (class MonadAff)
import Effect.Class (liftEffect)
import Effect.Class.Console (log)
import Effect.Exception (error)
import Element (replaceChildren)
import Halogen as H
import Halogen.Aff as HA
import Halogen.HTML as HH
import Halogen.HTML.Properties as HP
import Halogen.VDom.Driver (runUI)
import TW.TW (twMerge)
import Web.DOM.ParentNode (QuerySelector(..))
import Web.HTML.HTMLElement (HTMLElement, toElement)

main :: Effect Unit
main = HA.runHalogenAff do
  HA.awaitLoad
  html <- selectElement_ "html"
  -- clear any existing values and re render. 
  -- initial html served should contain references to all the expensive assets so they can
  -- take advantage of browser initial page loading optimiations
  liftEffect $ replaceChildren (toElement html) []
  -- add each child of html (i.e. - head, body as described in headContents) as its own component
  traverse_ (\elem -> runUI (rootComponent elem) unit html) rootContents

rootComponent :: ∀ a q i o m. HH.HTML (H.ComponentSlot a m Unit) Unit -> H.Component q i o m
rootComponent elem =
  H.mkComponent
    { initialState
    , render
    , eval: H.mkEval $ H.defaultEval { handleAction = pure }
    }
  where
  initialState _ = elem
  render state = state

rootContents =
  [ HH.head_
      [ HH.title_ [ HH.text "Hello World" ] ]
  , HH.body
      [ css "h-full leading-relaxed font-light text-lg text-warm-gray" ]
      [ HH.div
          [ css "h-full w-full flex justify-center items-center" ]
          [ HH.h1
              [ css "font-7xl font-black" ]
              [ HH.text "Hello World" ]
          ]
      ]
  ]

css :: ∀ r i. String -> HH.IProp (class :: String | r) i
css = HP.class_ <<< HH.ClassName

cssMerge :: ∀ r i. Array String -> HH.IProp (class :: String | r) i
cssMerge = css <<< twMerge

-- | Finds the element and throws if it cannot be found. Call after the document has loaded.
selectElement_ :: String -> Aff HTMLElement
selectElement_ name = do
  elem <- HA.selectElement (QuerySelector name)
  maybe (throwError <<< error $ "No elements matched the query selector `" <> name <> "`") pure elem
