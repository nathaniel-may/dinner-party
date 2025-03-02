#!/usr/bin/env stack
-- stack --resolver lts-21.24 --no-install-ghc --system-ghc runghc --package turtle

{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Concurrent (threadDelay)
import Control.Concurrent.Async (Async, async, forConcurrently_, mapConcurrently_, withAsync)
import Control.Exception (Exception (..), throwIO)
import Control.Monad (when)
import Control.Monad.Managed (MonadManaged)
import Data.ByteString (ByteString)
import Data.Foldable (traverse_)
import Data.List (foldl')
import Data.Maybe (catMaybes, fromMaybe, mapMaybe)
import Data.Text qualified as T
import Data.Text.IO qualified as T
import Options.Applicative
import System.Console.ANSI
import System.Exit (ExitCode (..))
import System.FilePath (replaceExtension)
import System.IO qualified as SYS
import Turtle

data App = Run Command | Test deriving (Read, Show, Eq)

data Command = Build | BuildRelease | Serve | Develop | Fmt | Install | Clean deriving (Read, Show, Eq)

newtype ScriptException = ScriptException ByteString deriving (Show)

instance Exception ScriptException

newtype TestingException = TestingException Text deriving (Show)

instance Exception TestingException

main :: IO ()
main = do
  app <- customExecParser (prefs showHelpOnEmpty) parser
  run app

run :: (MonadIO m) => App -> m ()
run =
  liftIO . \case
    Test -> mapConcurrently_ id tests
    Run cmd -> case cmd of
      Build -> sh build
      BuildRelease -> sh buildRelease
      Serve -> view $ do run (Run Build); pushd "dist" >> procs "npx" ["http-server", "-o", "-c-1"] empty
      Develop -> print "todo implement develop"
      Fmt -> procs "purs-tidy" ["format-in-place", "src/**/*.purs"] empty
      Install -> view $ procs "npm" ["install"] empty
      Clean -> traverse_ (\dir -> procs "rm" ["-rf", T.pack dir] empty) builtPaths

build :: Shell ()
build =
  let buildStyles =
        procs
          "npx"
          [ "@tailwindcss/cli",
            "-c",
            "tailwind.config.js",
            "-i",
            "./src/style.css",
            "-o",
            "./dist/style.css"
          ]
          empty
      -- calling npm run instead of spago bundle-app directly so spago can find `esbuild` in local node_modules
      buildJS = procs "npm" ["run", "spago-bundle-app"] empty
   in do
        view moveStuff
        run (Run Install)
        liftIO $ mapConcurrently_ view [buildStyles, buildJS]

-- optimizations for serving like minifying source, generating the initial html, and converting images to webp
buildRelease :: Shell ()
buildRelease = do
  sh moveStuff
  run (Run Install)
  liftIO $ mapConcurrently_ view [buildStyles, buildJS, buildImages, genHTML]
  where
    buildJS = procs "npm" ["run", "spago-bundle-app-release"] empty
    buildStyles =
      procs
        "npx"
        [ "tailwindcss",
          "-c",
          "tailwind.config.js",
          "-i",
          "./src/style.css",
          "-o",
          "./dist/style.css",
          "--minify"
        ]
        empty
    buildImages = do
      file <- ls "dist/images/"
      procs "cwebp" ["-quiet", "-q", "80", T.pack file, "-o", T.pack $ replaceExtension file "webp"] empty
    genHTML =
      -- ignoring handle because npm doesn't pass through termination signals. We'll kill via the OS
      liftIO $
        withAsync
          (sh $ pushd "dist" *> procs "npx" ["http-server", "-c-1", "--port", "8111"] empty)
          ( \_ -> do
              let line =
                    inproc
                      -- todo this path is mac specific, treat headless chrome as a dependency instead
                      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                      ["--headless=new", "--dump-dom", "http://127.0.0.1:8111"]
                      empty
              sh (output "index2.html" line)
              -- kill the server once all the lines have been dumped. npx doesn't passthrough termination signals,
              -- so we can't have nice things like process handles and information on when or if the process ended.
              sh $ procs "killall" ["http-server"] empty
              -- overwrite the old file with the new contents
              mv "index2.html" "index.html"
          )

moveStuff :: Shell ()
moveStuff = do
  -- dist is the directory the server serves from. wipe the previous build output and start over
  procs "rm" ["-rf", "dist"] empty
  mkdir "dist"
  -- copy whole dirs with their existing structure before moving individual files
  procs "cp" ["-r", "./assets/images", "dist"] empty
  -- move individual files
  procs "cp" ["./src/index.html", "dist"] empty
  procs "cp" ["./src/404.html", "dist"] empty
  -- copy all the favicons to the root of the server folder
  ls "./assets/favicon" >>= (`cp` "dist")

-- leading "./" and lack of trailing "/" is necessary to match out put of find
builtPaths :: [FilePath]
builtPaths =
  [ "./dist",
    "./node_modules",
    "./output",
    "./.spago",
    "./.stack-work"
  ]

builtPathsText :: [Text]
builtPathsText = map T.pack builtPaths

parser :: ParserInfo App
parser =
  info
    (subcommands <**> helper)
    (fullDesc <> progDesc "Tools for developing and deploying your application")
  where
    subcommands :: Parser App
    subcommands =
      subparser
        ( command "build" (info (pure $ Run Build) (progDesc "build without optimizations"))
            <> command "build-release" (info (pure $ Run BuildRelease) (progDesc "build with optimiations for deploying to production"))
            <> command "serve" (info (pure $ Run Serve) (progDesc "serve the web app"))
            <> command "develop" (info (pure $ Run Develop) (progDesc "serve and watch for source file changes to reload the page"))
            <> command "fmt" (info (pure $ Run Fmt) (progDesc "format *.purs files in place"))
            <> command "install" (info (pure $ Run Install) (progDesc "install build dependencies"))
            <> command "clean" (info (pure $ Run Clean) (progDesc "delete all build files"))
            <> command "test-scripts" (info (pure Test) (progDesc "test all the scripts in this file"))
        )

for :: (Functor f) => f a -> (a -> b) -> f b
for = flip fmap

for_ :: (Functor f) => f a -> (a -> b) -> f ()
for_ x f = void $ fmap f x

test :: (MonadIO m) => Text -> m a -> (a -> Bool) -> Text -> m ()
test name actual fExpected msg = do
  result <- fExpected <$> actual
  let errOutput = do
        T.putStrLn "‼️ TEST FAILED ‼️"
        throwIO (TestingException msg)
  if result
    then pure ()
    else liftIO $ do
      stdoutSupportsANSI <- hSupportsANSI SYS.stdout
      if stdoutSupportsANSI
        then do
          setSGR
            [ SetConsoleIntensity BoldIntensity,
              SetColor Foreground Vivid Red
            ]
          errOutput
          setSGR [Reset]
        else
          errOutput

-- todo write meaningful tests
tests :: [IO ()]
tests =
  [ test
      "build command parses"
      (handleParseResult $ execParserPure (prefs showHelpOnEmpty) parser ["build"])
      (== Run Build)
      "expected build input to produce Build result"
  ]
