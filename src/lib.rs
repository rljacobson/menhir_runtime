// The Rust runtime for Menhir.
// All the generated parsers run the function below that implements a classic
// LR(1) automaton with their generated parse table.

pub type Stack<YYType, State> = Vec<(State, YYType)>;

// The type of semantic actions.
// Some(ptr) = a flat (not closure) code pointer to the handler
// None = this is a start reduction. it's actually never executed but indicates
// an Accept action instead. Since NULL is not a valid function pointer in correct
// Rust code, this should be optimized to be just the size of a function pointer.
pub type SemAct<YYType, State> = Option<fn(State, &mut Stack<YYType, State>) -> State>;

// An action, i.e. an entry in the action table. Error, accept, reduce with a
// semantic function that takes the stack, modifie it and returns the next state
// or shift to state, discarding the current token.
pub enum Action<YYType, State> {
  Err,
  Reduce(SemAct<YYType, State>),
  Shift(State)
}

impl<T, U: Copy> Copy for Action<T, U> {}

impl<T, U: Clone> Clone for Action<T, U> {
  fn clone(&self) -> Self {
    match *self {
      Action::Err => Action::Err,
      Action::Reduce(act) => Action::Reduce(act),
      Action::Shift(ref shift) => Action::Shift(shift.clone())
    }
  }
}

pub trait LRParser {
  type Terminal: Copy;
  fn error() -> Self::Terminal;

  type State: Copy;
  type YYType;

  fn default_reduction(state: Self::State)
                       -> Option<SemAct<Self::YYType, Self::State>>;
  fn action(state: Self::State, token: Self::Terminal)
            -> Action<Self::YYType, Self::State>;
}

pub trait Lexer {
  type Location;
  type Token;
  type Error;

  fn input(&mut self) -> Result<(Self::Location, Self::Token), Self::Error>;
}

// Convert an iterator into a lexer.
pub struct IteratorLexer<Iter, Loc, Tok>
  where Iter: Iterator<Item = (Loc, Tok)> {
  iter: Iter,
  last_pos: Loc,
  marker: ::std::marker::PhantomData<(Loc, Tok)>
}

/// The parser reached EOF while it was still expecting input. This
/// error can only be raised when using an IteratorLexer as input.
#[derive(Clone, Copy, Debug)]
pub struct UnexpectedEof<Location>(pub Location);

impl<Iter, Loc, Tok> Lexer for IteratorLexer<Iter, Loc, Tok>
  where Loc: Clone, Iter: Iterator<Item = (Loc, Tok)> {
  type Location = Loc;
  type Token = Tok;
  type Error = UnexpectedEof<Self::Location>;

  fn input(&mut self) -> Result<(Loc, Tok), Self::Error> {
    match self.iter.next() {
      Some((pos, tok)) => {
        self.last_pos = pos.clone();
        Ok((pos, tok))
      }
      None => Err(UnexpectedEof(self.last_pos.clone()))
    }
  }
}

impl<Iter, Loc, Tok> IteratorLexer<Iter, Loc, Tok>
  where Loc: Default, Iter: Iterator<Item = (Loc, Tok)> {
  pub fn new(lex: Iter) -> Self {
    IteratorLexer {
      iter: lex,
      last_pos: Loc::default(),
      marker: ::std::marker::PhantomData
    }
  }
}

// Describe a possible entry point of the given parser.
pub trait EntryPoint<Parser: LRParser> {
  type Output;

  fn extract_output(stack: Stack<Parser::YYType, Parser::State>) -> Self::Output;
  fn initial() -> Parser::State;

  fn new<Lexer>(lex: Lexer)
                -> Result<ParserState<Lexer, Parser, Self>, Lexer::Error>
    where Lexer: self::Lexer,
          Lexer::Token: Into<(Parser::YYType, Parser::Terminal)>,
          Self: ::std::marker::Sized {
    new::<Lexer, Parser, Self>(lex)
  }

  fn run<Lexer>(lex: Lexer) -> Result<Self::Output, ParserError<Lexer>>
    where Lexer: self::Lexer,
          Lexer::Token: Into<(Parser::YYType, Parser::Terminal)>,
          Lexer::Location: Clone,
          Parser: LRErrors,
          Self: ::std::marker::Sized {
    run::<Lexer, Parser, Self>(lex)
  }
}

// A fatal (non-recoverable parsing error).
#[derive(Debug)]
pub enum ParserError<Lexer: self::Lexer> {
  // The parser encountered a syntax error that couldn't be recovered.
  SyntaxError(Lexer::Location, Option<&'static str>),

  // The lexer encountered an error, typically an IO error.
  LexerError(Lexer::Error)
}

// The result of a parsing process.
pub enum ParseResult<Output, Error, Fatal> {
  Success(Output),
  Error(Error),
  Fatal(Fatal)
}

// Creates a new parser from the given lexer.
pub fn new<Lexer, Parser, Entry>(mut lex: Lexer)
                                 -> Result<ParserState<Lexer, Parser, Entry>, Lexer::Error>
  where Lexer: self::Lexer,
        Lexer::Token: Into<(Parser::YYType, Parser::Terminal)>,
        Parser: LRParser,
        Entry: EntryPoint<Parser> {
  let state = Entry::initial();
  let stack = Vec::new();
  let (pos, tok) = lex.input()?;
  let (yylval, tok) = tok.into();
  Ok(ParserState {
    state: state, stack: stack, yylval: yylval,
    lookahead: tok, location: pos, lexer: lex,
    entry: ::std::marker::PhantomData
  })
}

// Creates a new parser and run it. Equivalent to
pub fn run<Lexer, Parser, Entry>(lex: Lexer)
                                 -> Result<Entry::Output, ParserError<Lexer>>
  where Lexer: self::Lexer,
        Lexer::Token: Into<(Parser::YYType, Parser::Terminal)>,
        Lexer::Location: Clone,
        Parser: LRParser + LRErrors,
        Entry: EntryPoint<Parser> {
  let state = match new::<_, _, Entry>(lex) {
    Ok(state) => state,
    Err(err) => return Err(ParserError::LexerError(err))
  };
  state.run()
}

// The state of the parser.
pub struct ParserState<Lexer, Parser, Entry>
  where Parser: LRParser,
        Lexer: self::Lexer,
        Entry: EntryPoint<Parser> {
  state: Parser::State,
  yylval: Parser::YYType,
  stack: Stack<Parser::YYType, Parser::State>,
  lookahead: Parser::Terminal,
  location: Lexer::Location,
  lexer: Lexer,
  entry: ::std::marker::PhantomData<Entry>
}

impl<Lexer, Parser, Entry> ParserState<Lexer, Parser, Entry>
  where Parser: LRParser,
        Lexer: self::Lexer,
        Lexer:: Token: Into<(Parser::YYType, Parser::Terminal)>,
        Entry: EntryPoint<Parser> {
  pub fn step(self) -> ParseResult<Entry::Output,
    ErrorState<Lexer, Parser, Entry>,
    ParserError<Lexer>> {
    let ParserState {
      mut state, mut yylval, mut stack,
      mut lookahead, mut location, mut lexer,
      entry: _
    } = self;

    'a: loop {
      match Parser::action(state, lookahead) {
        Action::Shift(shift) => {
          stack.push((state, yylval));
          state = shift;

          while let Some(red) = Parser::default_reduction(state) {
            match red {
              Some(code) => state = code(state, &mut stack),
              None => break 'a
            }
          }

          // discard
          let (pos, (nval, tok)) = match lexer.input() {
            Ok((pos, tok)) => (pos, tok.into()),
            Err(err) =>
              return ParseResult::Fatal(ParserError::LexerError(err))
          };
          lookahead = tok;
          location = pos;
          yylval = nval;
        }

        Action::Reduce(Some(reduce)) => {
          state = reduce(state, &mut stack);
          while let Some(red) = Parser::default_reduction(state) {
            match red {
              Some(code) => state = code(state, &mut stack),
              None => break 'a
            }
          }
        }

        Action::Reduce(None) => break,
        Action::Err => {
          return ParseResult::Error(ErrorState {
            state: ParserState {
              state: state, yylval: yylval, stack: stack,
              lookahead: lookahead, location: location, lexer: lexer,
              entry: ::std::marker::PhantomData
            }
          });
        }
      }
    }

    ParseResult::Success(Entry::extract_output(stack))
  }
}

impl<Lexer, Parser, Entry> ParserState<Lexer, Parser, Entry>
  where Parser: LRParser + LRErrors,
        Lexer: self::Lexer,
        Lexer::Token: Into<(Parser::YYType, Parser::Terminal)>,
        Lexer::Location: Clone,
        Entry: EntryPoint<Parser> {
  pub fn run(mut self) -> Result<Entry::Output, ParserError<Lexer>> {
    loop {
      match self.step() {
        ParseResult::Success(out) => return Ok(out),
        ParseResult::Error(mut err_state) => {
          let (last_pos, last_err) = err_state.report_error();
          loop {
            match err_state.try_recover() {
              ParseResult::Success(ok_state) => {
                self = ok_state;
                break;
              }
              ParseResult::Fatal(()) =>
                return Err(ParserError::SyntaxError(
                  last_pos, last_err
                )),
              ParseResult::Error(new_err_state) =>
                err_state = new_err_state
            }
          }
        }
        ParseResult::Fatal(err) => return Err(err)
      }
    }
  }
}

// The state of the parser after a recoverable error.
pub struct ErrorState<Lexer, Parser, Entry>
  where Parser: LRParser,
        Lexer: self::Lexer,
        Entry: EntryPoint<Parser> {
  state: ParserState<Lexer, Parser, Entry>
}

// The result of an error recovery operation.
pub type RecoveryResult<Lexer, Parser, Entry> =
ParseResult<ParserState<Lexer, Parser, Entry>,
  ErrorState<Lexer, Parser, Entry>,
  ()>;

impl<Parser, Lexer, Entry> ErrorState<Lexer, Parser, Entry>
  where Parser: LRParser + LRErrors,
        Lexer: self::Lexer,
        Lexer::Location: Clone,
        Entry: EntryPoint<Parser> {
  // Give a proper error message to describe the situation in which the
  // parser entered error-handling mode.
  pub fn report_error(&self) -> (Lexer::Location, Option<&'static str>) {
    (self.state.location.clone(), Parser::message(self.state.state))
  }

  pub fn try_recover(mut self) -> RecoveryResult<Lexer, Parser, Entry> {
    if let Action::Err = Parser::action(self.state.state, Parser::error()) {
      return match self.state.stack.pop() {
        Some((state, _)) => {
          self.state.state = state;
          ParseResult::Error(self)
        }
        None => ParseResult::Fatal(())
      }
    }
    ParseResult::Success(
      ParserState {
        lookahead: Parser::error(),
        .. self.state
      }
    )
  }
}

// Types that can convert errors to a detailed message.
pub trait LRErrors: LRParser {
  fn message(state: Self::State) -> Option<&'static str>;
}
