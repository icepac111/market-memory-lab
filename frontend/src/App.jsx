import PrivateCaseViewer from './PrivateCaseViewer.jsx'
import { useState } from 'react'
import './App.css'

const initialTitle = 'Event study preparation'

const researchSteps = [
  {
    id: 'question',
    title: 'Define the question',
    detail: 'Specify an observable claim, not just an interesting story.',
  },
  {
    id: 'source',
    title: 'Verify the event source',
    detail: 'Record a source URL and distinguish publication from occurrence.',
  },
  {
    id: 'data',
    title: 'Specify the dataset',
    detail: 'Identify the instrument, frequency, timezone, and adjustments.',
  },
  {
    id: 'challenge',
    title: 'Plan the challenge',
    detail: 'Write down a benchmark, confounds, and what could change your mind.',
  },
]

const evidenceCards = [
  {
    number: '01',
    label: 'Observation',
    title: 'Evidence before narrative.',
    detail: 'No empirical observation recorded. This workspace contains demonstration content only.',
    color: 'mint',
  },
  {
    number: '02',
    label: 'Limitation',
    title: 'Unknowns stay visible.',
    detail: 'No market series, verified event, benchmark, or analysis has been attached.',
    color: 'purple',
  },
  {
    number: '03',
    label: 'Next experiment',
    title: 'Make the question testable.',
    detail: 'Attach a verifiable event source, define the dataset, and write an explicit hypothesis.',
    color: 'peach',
  },
]

export default function App() {
  const [title, setTitle] = useState(initialTitle)
  const [instrument, setInstrument] = useState('')
  const [notes, setNotes] = useState('')
  const [completed, setCompleted] = useState([])
  const [focusMode, setFocusMode] = useState(false)
  const [orbitRotation, setOrbitRotation] = useState(0)
  const [orbitPalette, setOrbitPalette] = useState(0)
  const [orbitPaused, setOrbitPaused] = useState(false)
  const [showAbout, setShowAbout] = useState(false)

  function resetOrbit() {
    setOrbitRotation(0)
    setOrbitPalette(0)
    setOrbitPaused(false)
  }

  const completion = Math.round(
    (completed.length / researchSteps.length) * 100,
  )

  function toggleStep(id) {
    setCompleted((previous) =>
      previous.includes(id)
        ? previous.filter((step) => step !== id)
        : [...previous, id],
    )
  }

  function resetDraft() {
    setTitle(initialTitle)
    setInstrument('')
    setNotes('')
    setCompleted([])
    setFocusMode(false)
  }

  return (
    <div className={`app-shell ${focusMode ? 'focus-mode' : ''}`}>
      <a className="skip-link" href="#workspace">Skip to workspace</a>

      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="Market Memory Lab workspace">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span>MARKET MEMORY<span className="brand-light"> / LAB</span></span>
        </a>
        <span className="local-badge">
          <span className="status-dot" aria-hidden="true" />
          LOCAL DEMO
        </span>
      </header>

      <main id="workspace">
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <p className="eyebrow">A WORKSPACE FOR BETTER QUESTIONS</p>
            <h1 id="page-title">
              Markets move.<br />
              <span>Question the structure.</span>
            </h1>
            <p className="hero-description">
              Turn a market story into an inspectable research case.
              Keep the question sharp, the inputs traceable, and the
              unknowns in plain sight.
            </p>
            <div className="hero-tags">
              <span>01 / DEFINE</span>
              <span>02 / INSPECT</span>
              <span>03 / CHALLENGE</span>
            </div>
          </div>

          <section className="orbit-playground" aria-label="Interactive decorative orbit">
            <div
              className={`orbital-art palette-${orbitPalette} ${orbitPaused ? 'orbit-paused' : ''}`}
              aria-hidden="true"
            >
              <div
                className="orbital-scene"
                style={{ transform: `rotate(${orbitRotation}deg)` }}
              >
                <div className="orbit orbit-one" />
                <div className="orbit orbit-two" />
                <div className="orbit orbit-three" />
                <div className="planet" />
              </div>
              <span className="orbit-caption">QUESTION / TEST / REVISE</span>
            </div>

            <div className="orbit-controls" role="group" aria-label="Orbit appearance controls">
              <button
                className="orbit-button"
                type="button"
                onClick={() => setOrbitRotation((previous) => previous + 90)}
              >
                Spin it ↻
              </button>
              <button
                className="orbit-button"
                type="button"
                onClick={() => setOrbitPalette((previous) => (previous + 1) % 3)}
              >
                Remix colors
              </button>
              <button
                className="orbit-button"
                type="button"
                aria-pressed={orbitPaused}
                onClick={() => setOrbitPaused((previous) => !previous)}
              >
                {orbitPaused ? 'Play float' : 'Pause float'}
              </button>
              <button
                className="orbit-button"
                type="button"
                onClick={resetOrbit}
              >
                Reset orbit
              </button>
            </div>
            <p className="orbit-disclaimer">
              A little room to play. Decorative art — not a scientific simulation.
            </p>
          </section>
        </section>

        <section className="researcher-strip" aria-label="Researcher profile">
          <div>
            <p className="eyebrow">BUILT BY A RESEARCHER, FOR BETTER RESEARCH</p>
            <p className="researcher-name">Pranshu Mihirbhai Jayswal</p>
          </div>
          <div className="researcher-actions">
            <button
              className="button button-secondary"
              type="button"
              aria-expanded={showAbout}
              aria-controls="researcher-details"
              onClick={() => setShowAbout((previous) => !previous)}
            >
              {showAbout ? 'Close profile' : 'Meet the researcher'}
            </button>
            <a
              className="button button-secondary research-link"
              href="https://github.com/icepac111/market-memory-lab"
              target="_blank"
              rel="noopener noreferrer"
            >
              Explore the code ↗
            </a>
          </div>
          <div id="researcher-details" hidden={!showAbout}>
            <p>
              Financial Technology PhD student at Worcester Polytechnic Institute
              and creator of Market Memory Lab. Background in data science,
              Python, statistical inference, time-series analysis, and
              reproducible research workflows.
            </p>
            <p>
              Research interests include point-in-time financial information,
              transparent baselines, temporal validation, and making
              analytical limitations visible.
            </p>
          </div>
        </section>

        <aside className="demo-banner">
          <span className="banner-icon" aria-hidden="true">!</span>
          <p>
            The draft editor below is fictional. The separate case viewer displays locally selected research outputs.
            Draft edits stay in this tab’s memory and reset on refresh.
          </p>
        </aside>

        <PrivateCaseViewer />

        <section className="workspace-heading" aria-labelledby="case-heading">
          <div>
            <p className="eyebrow">YOUR RESEARCH DESK</p>
            <h2 id="case-heading">Evidence workbench</h2>
          </div>
          <button
            className="button button-secondary"
            type="button"
            aria-pressed={focusMode}
            onClick={() => setFocusMode((previous) => !previous)}
          >
            {focusMode ? 'Show full workspace' : 'Enter focus mode'}
          </button>
        </section>

        <div className="workspace-grid">
          <article className="panel case-panel">
            <div className="panel-topline">
              <span className="mono-label">DEMO-001 / CASE DRAFT</span>
              <span className="hypothesis-badge">Untested hypothesis</span>
            </div>

            <h3 className="case-title">{title.trim() || 'Untitled case'}</h3>
            <p className="case-description">
              What inputs would we need to examine a market response
              around a documented event?
            </p>

            <div className="metadata">
              <div>
                <span>INSTRUMENT DRAFT</span>
                <p>{instrument.trim() || 'Not selected'}</p>
              </div>
              <div>
                <span>VERIFIED SOURCE</span>
                <p>Not attached</p>
              </div>
              <div>
                <span>EVENT TIMESTAMP</span>
                <p>Not supplied</p>
              </div>
            </div>

            <div className="editor">
              <div className="section-caption">
                <span className="tiny-light" aria-hidden="true" />
                LIVE DRAFT EDITOR
              </div>

              <label htmlFor="case-title">Case title</label>
              <input
                id="case-title"
                value={title}
                maxLength={120}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Give your question a name"
              />

              <label htmlFor="instrument">Instrument label — unverified</label>
              <input
                id="instrument"
                value={instrument}
                maxLength={60}
                onChange={(event) => setInstrument(event.target.value)}
                placeholder="Type a label; this does not fetch or validate data"
              />

              <label htmlFor="research-notes">Research scratchpad</label>
              <textarea
                id="research-notes"
                value={notes}
                maxLength={2000}
                onChange={(event) => setNotes(event.target.value)}
                placeholder="What would convince you that your hypothesis is wrong?"
                rows={4}
              />
              <div className="editor-footer">
                <span>{notes.length} / 2000 characters · not saved</span>
                <button
                  className="button button-reset"
                  type="button"
                  onClick={resetDraft}
                >
                  Reset demo
                </button>
              </div>
            </div>
          </article>

          <aside className="panel checklist-panel" aria-labelledby="checklist-title">
            <p className="eyebrow">PROCESS, NOT PROOF</p>
            <h3 id="checklist-title">Preparation checklist</h3>
            <p className="muted checklist-description">
              Self-reported steps. Ticking a box does not verify the work.
            </p>

            <div className="progress-heading">
              <span>{completed.length} of {researchSteps.length} marked</span>
              <span>{completion}%</span>
            </div>
            <progress
              value={completed.length}
              max={researchSteps.length}
              aria-label="Self-reported preparation completion"
            />

            <div className="checklist">
              {researchSteps.map((step) => (
                <label
                  className={`checklist-item ${completed.includes(step.id) ? 'is-complete' : ''}`}
                  key={step.id}
                >
                  <input
                    type="checkbox"
                    checked={completed.includes(step.id)}
                    onChange={() => toggleStep(step.id)}
                  />
                  <span>
                    <span className="step-title">{step.title}</span>
                    <span className="step-detail">{step.detail}</span>
                  </span>
                </label>
              ))}
            </div>

            <div className="research-rule">
              <span className="mono-label">LAB RULE / 001</span>
              <p>A compelling story is a starting point. Not a result.</p>
            </div>
          </aside>
        </div>

        <section className="evidence-grid" aria-label="Evidence and research limitations">
          {evidenceCards.map((card) => (
            <article className={`evidence-card ${card.color}`} key={card.number}>
              <div className="card-topline">
                <span>{card.label}</span>
                <span>{card.number}</span>
              </div>
              <h3>{card.title}</h3>
              <p>{card.detail}</p>
            </article>
          ))}
        </section>

        <footer className="page-footer">
          <span>MARKET MEMORY LAB / EXPERIMENTAL INTERFACE</span>
          <span>No live feed. Historical results loaded from a local file. No investment advice.</span>
        </footer>
      </main>
    </div>
  )
}
