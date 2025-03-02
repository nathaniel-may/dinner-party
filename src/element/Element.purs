module Element where

import Prelude
import Effect (Effect)
import Web.DOM (Element)

foreign import replaceChildren :: Element -> Array Element -> Effect Unit
foreign import outerHTML :: Element -> Effect String