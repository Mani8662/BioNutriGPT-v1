import React, { useEffect, useRef, useState } from "react";
import {
  Leaf,
  LayoutDashboard,
  UserRound,
  HeartPulse,
  SlidersHorizontal,
  Utensils,
  MessageCircle,
  FileText,
  ChevronDown,
  Check,
  Plus,
  Menu,
  X,
  Send,
  History,
  Download,
  RefreshCw,
  ShieldCheck,
  Sun,
  Droplets,
  CheckCircle2,
  HelpCircle,
  LogOut,
  Flame,
  CalendarDays,
} from "lucide-react";
import { api } from "./api";
import { Button, Field, Select, Empty, Busy } from "./ui";
import Profile from "./Profile";
import { MealPlan, Report } from "./PlanViews";
import {
  FEATURE_DEFINITIONS,
  CANONICAL_SAMPLE_FEATURES,
  CANONICAL_SAMPLE_METADATA,
} from "./data/featureDefinitions";
import { downloadFile, splitList, validateProfile } from "./utils";
import type {
  PatientMetadata,
  ScreeningResult,
  MealPlanResult,
  ClinicalReport,
  MealPlanPreferences,
  ChatResponse,
  PlanSessionRecord,
} from "./types/api";

const pages = [
  { id: "overview", name: "Overview", icon: LayoutDashboard },
  { id: "profile", name: "My profile", icon: UserRound },
  { id: "insights", name: "Health insights", icon: HeartPulse },
  { id: "preferences", name: "Food preferences", icon: SlidersHorizontal },
  { id: "meals", name: "My meal plan", icon: Utensils },
  { id: "assistant", name: "Plan assistant", icon: MessageCircle },
  { id: "reports", name: "Reports", icon: FileText },
] as const;
type Page = (typeof pages)[number]["id"];
const blankMeta: PatientMetadata = {
  renal_disease: null,
  pregnant: null,
  lactating: null,
  weight_kg: NaN,
  height_cm: NaN,
  age_years: NaN,
  is_female: false,
  activity_level: "sedentary",
};
const initialPrefs: MealPlanPreferences = {
  dietary_preference: "veg_nonveg_combo",
  allergies: [],
  excluded_food_ids: [],
  disliked_terms: [],
  max_active_dishes: null,
};
const validPage = (v: string): v is Page => pages.some((p) => p.id === v);
export default function App() {
  const [page, setPage] = useState<Page>(() => {
      const p = location.hash.slice(1);
      return validPage(p) ? p : "overview";
    }),
    [mobile, setMobile] = useState(false),
    [account, setAccount] = useState(false),
    [help, setHelp] = useState(false);
  const [values, setValues] = useState<string[]>(
      Array(FEATURE_DEFINITIONS.length).fill(""),
    ),
    [meta, setMeta] = useState<PatientMetadata>({ ...blankMeta }),
    [sample, setSample] = useState(false),
    [reviewed, setReviewed] = useState(false);
  const [screening, setScreening] = useState<ScreeningResult | null>(null),
    [plan, setPlan] = useState<MealPlanResult | null>(null),
    [report, setReport] = useState<ClinicalReport | null>(null),
    [prefs, setPrefs] = useState<MealPlanPreferences>({ ...initialPrefs }),
    [allergies, setAllergies] = useState(""),
    [dislikes, setDislikes] = useState(""),
    [prefReview, setPrefReview] = useState(false);
  const [session, setSession] = useState<string | null>(null),
    [history, setHistory] = useState<PlanSessionRecord | null>(null),
    [showHistory, setShowHistory] = useState(false),
    [turns, setTurns] = useState<
      Array<{
        question: string;
        response: ChatResponse;
        applied?: boolean;
        revision: number;
      }>
    >([]),
    [question, setQuestion] = useState(""),
    [naturalChat, setNaturalChat] = useState(false),
    [flagged, setFlagged] = useState(false);
  const [busy, setBusy] = useState(""),
    [error, setError] = useState(""),
    [toast, setToast] = useState(""),
    [connected, setConnected] = useState<boolean | null>(null),
    [water, setWater] = useState(() => {
      try {
        return Math.max(
          0,
          Math.min(
            20,
            Number(
              localStorage.getItem(
                "bionutri-water-" + new Date().toDateString(),
              ),
            ) || 0,
          ),
        );
      } catch {
        return 0;
      }
    });
  const lock = useRef(false),
    revision = useRef(0),
    main = useRef<HTMLElement>(null),
    dialog = useRef<HTMLDialogElement>(null),
    endOfChat = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const fn = () => {
      const p = location.hash.slice(1);
      if (validPage(p)) setPage(p);
    };
    window.addEventListener("hashchange", fn);
    api
      .health()
      .then((h) => setConnected(h.pipeline_loaded))
      .catch(() => setConnected(false));
    return () => window.removeEventListener("hashchange", fn);
  }, []);
  useEffect(() => {
    document.title = `${pages.find((p) => p.id === page)?.name} | BioNutriGPT`;
    setMobile(false);
    setAccount(false);
    main.current?.focus();
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, [page]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 4500);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    if (help) dialog.current?.showModal();
    else dialog.current?.close();
  }, [help]);
  useEffect(() => {
    endOfChat.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [turns]);
  function go(p: Page) {
    location.hash = p;
    setPage(p);
    setError("");
  }
  async function act(label: string, fn: () => Promise<void>) {
    if (lock.current) return;
    lock.current = true;
    setBusy(label);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      lock.current = false;
      setBusy("");
    }
  }
  function invalidate() {
    revision.current++;
    setReviewed(false);
    setScreening(null);
    setPlan(null);
    setReport(null);
    setSession(null);
    setHistory(null);
    setShowHistory(false);
    setTurns([]);
    setQuestion("");
    setFlagged(false);
  }
  function updateMeta(m: PatientMetadata) {
    invalidate();
    setMeta(m);
    setValues((old) =>
      old.map((v, i) =>
        FEATURE_DEFINITIONS[i].name === "RIDAGEYR"
          ? Number.isNaN(m.age_years)
            ? ""
            : String(m.age_years)
          : FEATURE_DEFINITIONS[i].name === "SEX_FEMALE"
            ? String(Number(m.is_female))
            : v,
      ),
    );
  }
  function updateValue(i: number, v: string) {
    invalidate();
    setValues((old) => old.map((x, j) => (i === j ? v : x)));
    const key = FEATURE_DEFINITIONS[i].name;
    if (key === "RIDAGEYR")
      setMeta((old) => ({ ...old, age_years: v === "" ? NaN : Number(v) }));
    if (key === "SEX_FEMALE" && v !== "")
      setMeta((old) => ({ ...old, is_female: v === "1" }));
  }
  function example() {
    if (
      screening &&
      !window.confirm("Replace the current profile and clear its results?")
    )
      return;
    invalidate();
    setValues(CANONICAL_SAMPLE_FEATURES.map(String));
    setMeta({ ...CANONICAL_SAMPLE_METADATA });
    setSample(true);
    setToast("Example profile loaded. Review the values before continuing.");
  }
  function importText(text: string) {
    act("Reading your profile…", async () => {
      const p = await api.parse(text);
      if (
        !p.is_valid ||
        p.features.length !== FEATURE_DEFINITIONS.length ||
        p.features.some((v) => !Number.isFinite(v))
      )
        throw new Error(
          "This file is not a complete supported profile. Check the file and try again.",
        );
      invalidate();
      setValues(p.features.map(String));
      const age = FEATURE_DEFINITIONS.findIndex((f) => f.name === "RIDAGEYR"),
        sex = FEATURE_DEFINITIONS.findIndex((f) => f.name === "SEX_FEMALE");
      setMeta({
        ...blankMeta,
        ...p.metadata,
        age_years: p.metadata.age_years ?? p.features[age],
        is_female: p.metadata.is_female ?? Boolean(p.features[sex]),
      });
      setSample(false);
      setToast(
        "Profile imported. Review your details and measurements before continuing.",
      );
    });
  }
  function importDocument(file: File) {
    act("Analyzing report and extracting lab biomarkers…", async () => {
      const p = await api.uploadDocument(file);
      if (
        !p.is_valid ||
        p.features.length !== FEATURE_DEFINITIONS.length ||
        p.features.some((v) => !Number.isFinite(v))
      )
        throw new Error(
          "Could not extract a valid 87-feature profile from this document. Please check the file.",
        );
      invalidate();
      setValues(p.features.map(String));
      const age = FEATURE_DEFINITIONS.findIndex((f) => f.name === "RIDAGEYR"),
        sex = FEATURE_DEFINITIONS.findIndex((f) => f.name === "SEX_FEMALE");
      setMeta({
        ...blankMeta,
        ...p.metadata,
        age_years: p.metadata.age_years ?? p.features[age],
        is_female: p.metadata.is_female ?? Boolean(p.features[sex]),
      });
      setSample(false);
      setToast(
        p.summary || "Document parsed and 87-feature clinical profile extracted successfully.",
      );
    });
  }
  function screen() {
    const message = validateProfile(values, meta);
    if (message) {
      setError(message);
      return;
    }
    if (!reviewed) {
      setError("Please confirm that you have reviewed your profile.");
      return;
    }
    act("Preparing your health insights…", async () => {
      const s = await api.screen(values.map(Number), meta);
      revision.current++;
      setScreening(s);
      setPlan(null);
      setReport(null);
      setSession(null);
      setHistory(null);
      setTurns([]);
      setFlagged(false);
      setConnected(true);
      go("insights");
    });
  }
  function changedPrefs() {
    setPrefReview(false);
  }
  function generate() {
    if (!screening || !screening.certified_for_stage3) {
      setError("Complete your health assessment before creating a meal plan.");
      return;
    }
    if (!prefReview) {
      setError("Please review and confirm your food preferences.");
      return;
    }
    act("Putting your meal plan together…", async () => {
      const next = {
        ...prefs,
        allergies: splitList(allergies),
        disliked_terms: splitList(dislikes),
      };
      const p = await api.plan(screening.dossier, next);
      setPrefs(next);
      revision.current++;
      setPlan(p);
      setReport(null);
      setSession(null);
      setHistory(null);
      setTurns([]);
      setFlagged(false);
      go("meals");
      if (p.accepted) setToast("Your meal plan is ready.");
    });
  }
  async function ensureSession() {
    if (session) return session;
    if (!plan?.accepted) throw new Error("Create a meal plan first.");
    const s = await api.session(plan.dossier);
    setSession(s.session_id);
    return s.session_id;
  }
  function send(text: string) {
    if (!text.trim() || !plan?.accepted) return;
    act("Your assistant is reviewing your question…", async () => {
      const sid = await ensureSession();
      const r = await api.chat(text.trim(), plan.dossier, sid, naturalChat);
      setFlagged((prev) => prev || !!r.plan_flagged_unsuitable);
      setTurns((t) => [
        ...t,
        { question: text.trim(), response: r, revision: revision.current },
      ]);
      setQuestion("");
      if (r.intent === "rollback" && r.plan_version) {
        const restored = await api.rollback(sid, r.plan_version);
        applyPlan(restored.meal_plan, sid);
        setToast("The previous version has been restored.");
      }
      setHistory(null);
    });
  }
  function applyPlan(p: MealPlanResult, sid?: string | null) {
    revision.current++;
    setPlan(p);
    setReport(null);
    setFlagged(
      (p.dossier?.stage3_discrete_meal_synthesis
        ?.flagged_unsuitable_for_allergies?.length || 0) > 0,
    );
    if (sid) setSession(sid);
    setHistory(null);
  }
  function accept(index: number) {
    const turn = turns[index];
    if (
      !plan ||
      !turn.response.proposed_meal_plan ||
      turn.revision !== revision.current
    )
      return;
    act("Updating your meal plan…", async () => {
      const r = await api.accept(
        plan.dossier,
        turn.response.proposed_meal_plan!,
        session,
      );
      applyPlan(r.meal_plan, r.session_id);
      setTurns((t) =>
        t.map((x, i) => (i === index ? { ...x, applied: true } : x)),
      );
      setToast(
        "Your meal plan has been updated. Your report can now be regenerated.",
      );
    });
  }
  function loadHistory() {
    act("Loading your plan history…", async () => {
      const sid = await ensureSession();
      setHistory(await api.history(sid));
      setShowHistory(true);
    });
  }
  function restore(v: number) {
    if (
      !session ||
      !window.confirm(
        `Restore plan version ${v}? Your current report will be cleared.`,
      )
    )
      return;
    act("Restoring your meal plan…", async () => {
      const r = await api.rollback(session, v);
      applyPlan(r.meal_plan, r.session_id);
      setHistory(await api.history(session));
      setToast(`Version ${v} restored.`);
    });
  }
  function makeReport() {
    if (!plan?.accepted) return;
    act("Preparing your personal report…", async () => {
      setReport(await api.report(plan.dossier));
      setToast("Your report is ready to download or print.");
    });
  }
  function reset() {
    if (
      busy ||
      !window.confirm(
        "Clear this profile, plan, and report from this browser session? Saved server history will not be deleted.",
      )
    )
      return;
    invalidate();
    setValues(Array(FEATURE_DEFINITIONS.length).fill(""));
    setMeta({ ...blankMeta });
    setSample(false);
    setPrefs({ ...initialPrefs });
    setAllergies("");
    setDislikes("");
    setPrefReview(false);
    go("overview");
    setToast("Workspace cleared. You can start a new profile.");
  }
  function waterChange(n: number) {
    const v = Math.max(0, Math.min(20, water + n));
    setWater(v);
    try {
      localStorage.setItem(
        "bionutri-water-" + new Date().toDateString(),
        String(v),
      );
    } catch {}
  }
  const title = pages.find((p) => p.id === page)!.name;
  const subtitle: Record<Page, string> = {
    overview: "A little clarity. A healthier everyday.",
    profile: "Your details, one simple section at a time.",
    insights: "Understand your starting point before taking the next step.",
    preferences: "Make room for the foods and routines you enjoy.",
    meals: "A daily plan that brings your preferences together.",
    assistant:
      "Questions, food swaps, and small adjustments. All in one conversation.",
    reports: "Your nutrition story, ready to read and share.",
  };
  const nextPage: Page = !screening
    ? "profile"
    : !plan?.accepted
      ? "preferences"
      : "meals";
  return (
    <div className="app">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      {mobile && (
        <button
          className="nav-overlay"
          aria-label="Close navigation"
          onClick={() => setMobile(false)}
        />
      )}
      <aside className={`sidebar ${mobile ? "open" : ""}`}>
        <button className="brand" onClick={() => go("overview")}>
          <span>
            <Leaf size={23} />
          </span>
          BioNutri<b>GPT</b>
        </button>
        <span className="nav-caption">YOUR PERSONAL SPACE</span>
        <nav aria-label="Main navigation">
          {pages.map((p) => (
            <button
              key={p.id}
              className={page === p.id ? "active" : ""}
              aria-current={page === p.id ? "page" : undefined}
              onClick={() => go(p.id)}
            >
              <p.icon size={19} />
              {p.name}
              {p.id === "meals" && plan?.accepted && (
                <Check size={14} className="nav-check" />
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-support">
          <span className="support-icon">
            <Leaf size={24} />
          </span>
          <h3>
            Better habits.
            <br />
            Your own pace.
          </h3>
          <p>You don’t have to change everything at once.</p>
          <button onClick={() => go(nextPage)}>Continue my journey</button>
        </div>
        <button className="help-link" onClick={() => setHelp(true)}>
          <HelpCircle size={18} />
          Help & getting started
        </button>
        <div className="sidebar-foot">
          <span className="avatar">Y</span>
          <div>
            <strong>Your workspace</strong>
            <small>Personal nutrition</small>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button mobile-toggle"
              aria-label="Open navigation"
              onClick={() => setMobile(true)}
            >
              <Menu />
            </button>
            <span>My workspace</span>
            <span className="slash">/</span>
            <strong>{title}</strong>
          </div>
          <div className="header-right">
            <span className="date-label">
              <CalendarDays size={15} />
              {new Date().toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </span>
            <div className="account-menu">
              <button
                className="account-trigger"
                aria-expanded={account}
                aria-label="Workspace menu"
                onClick={() => setAccount(!account)}
              >
                <span className="avatar">Y</span>
                <ChevronDown size={15} />
              </button>
              {account && (
                <div className="dropdown">
                  <button onClick={() => go("profile")}>
                    <UserRound size={16} />
                    My profile
                  </button>
                  <button
                    onClick={() => {
                      setHelp(true);
                      setAccount(false);
                    }}
                  >
                    <HelpCircle size={16} />
                    Help
                  </button>
                  <button
                    disabled={!!busy}
                    onClick={() => {
                      reset();
                      setAccount(false);
                    }}
                  >
                    <LogOut size={16} />
                    Start a new profile
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        <main id="main" ref={main} tabIndex={-1} className="content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">
                {page === "overview"
                  ? "YOUR WELLNESS, SIMPLIFIED"
                  : "PERSONAL NUTRITION"}
              </span>
              <h1>
                {page === "overview"
                  ? "Make space for a healthier you."
                  : title}
              </h1>
              <p>{subtitle[page]}</p>
            </div>
            {page === "overview" && (
              <Button variant="secondary" onClick={() => go("profile")}>
                <UserRound size={16} />
                {screening ? "Edit my profile" : "Set up my profile"}
              </Button>
            )}
          </div>
          {sample && (
            <div className="sample-note">
              <FlaskIcon />
              Example profile in use. These results are not your personal health
              results.
            </div>
          )}
          {error && (
            <div className="alert" role="alert">
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={18} />
              </button>
            </div>
          )}
          {busy && <Busy text={busy} />}
          <div className="page-body" key={page}>
            {page === "overview" && (
              <>
                <section className="hero">
                  <div className="hero-copy">
                    <span className="hero-label">
                      <Sun size={15} />A FRESH START, EVERY DAY
                    </span>
                    <h2>
                      Good food.
                      <br />
                      <span>A plan that feels like you.</span>
                    </h2>
                    <p>
                      Bring your health, food preferences, and everyday routine
                      into one thoughtful plan.
                    </p>
                    <Button onClick={() => go(nextPage)}>
                      {screening
                        ? plan?.accepted
                          ? "View my meal plan"
                          : "Choose my food preferences"
                        : "Start my personal plan"}
                      <Plus size={17} />
                    </Button>
                    <div className="hero-proof">
                      <span>
                        <Check size={14} />
                        Your preferences
                      </span>
                      <span>
                        <Check size={14} />
                        Familiar foods
                      </span>
                      <span>
                        <Check size={14} />
                        Clear next steps
                      </span>
                    </div>
                  </div>
                  <div className="hero-visual" aria-hidden="true">
                    <div className="orbital o1" />
                    <div className="orbital o2" />
                    <div className="plate">
                      <div className="plate-inner">
                        <div className="salad-leaf l1" />
                        <div className="salad-leaf l2" />
                        <div className="salad-leaf l3" />
                        <div className="salad-leaf l4" />
                        <div className="salad-leaf l5" />
                        <div className="grain g1" />
                        <div className="grain g2" />
                        <div className="grain g3" />
                        <div className="tomato t1" />
                        <div className="tomato t2" />
                        <div className="tomato t3" />
                        <div className="cucumber c1" />
                        <div className="cucumber c2" />
                        <div className="cucumber c3" />
                      </div>
                    </div>
                    <div className="floating-note">
                      <span>
                        <Leaf size={20} />
                      </span>
                      <div>
                        A little balance.<strong>A better everyday.</strong>
                      </div>
                    </div>
                  </div>
                </section>
                <div className="section-heading">
                  <div>
                    <h2>Your journey, made simple.</h2>
                    <p>One step at a time. Everything in its place.</p>
                  </div>
                  <span className="pill">
                    {screening ? (plan?.accepted ? 3 : 1) : 0} of 3 steps
                    complete
                  </span>
                </div>
                <div className="journey-grid">
                  {[
                    {
                      p: "profile" as Page,
                      icon: UserRound,
                      title: "Get to know you",
                      text: "Your health profile and everyday habits.",
                      done: !!screening,
                      button: screening ? "Review profile" : "Complete profile",
                    },
                    {
                      p: "preferences" as Page,
                      icon: SlidersHorizontal,
                      title: "Make it personal",
                      text: "Your food choices, allergies, and preferences.",
                      done: !!plan?.accepted,
                      button: "Choose preferences",
                    },
                    {
                      p: "meals" as Page,
                      icon: Utensils,
                      title: "Find your rhythm",
                      text: "Meals, portions, and a plan for your day.",
                      done: !!plan?.accepted,
                      button: plan?.accepted
                        ? "Open meal plan"
                        : "Explore next steps",
                    },
                  ].map((s) => (
                    <article className="card journey-card" key={s.p}>
                      <div className="journey-card-top">
                        <span className="icon-tile">
                          <s.icon size={22} />
                        </span>
                        {s.done ? (
                          <span className="done-label">
                            <CheckCircle2 size={15} />
                            Complete
                          </span>
                        ) : (
                          <span className="muted mini">Ready when you are</span>
                        )}
                      </div>
                      <h3>{s.title}</h3>
                      <p>{s.text}</p>
                      <button className="text-link" onClick={() => go(s.p)}>
                        {s.button}
                      </button>
                    </article>
                  ))}
                </div>
                <div className="overview-bottom">
                  <section className="card water-card">
                    <div>
                      <span className="eyebrow">A SMALL DAILY HABIT</span>
                      <h3>Make time for a water break.</h3>
                      <p className="muted">
                        Track what you drink, one glass at a time.
                      </p>
                      <div className="water-controls">
                        <Button
                          variant="secondary"
                          disabled={!water}
                          onClick={() => waterChange(-1)}
                        >
                          Undo
                        </Button>
                        <Button
                          disabled={water >= 20}
                          onClick={() => waterChange(1)}
                        >
                          <Plus size={16} />
                          Add a glass
                        </Button>
                      </div>
                    </div>
                    <div className="water-meter">
                      <Droplets size={25} />
                      <strong>
                        {(water * 0.25).toFixed(2)}
                        <small>L</small>
                      </strong>
                      <span>{water} × 250 ml today</span>
                    </div>
                  </section>
                  <section className="card personal-note">
                    <span className="icon-tile">
                      <ShieldCheck size={22} />
                    </span>
                    <h3>Your health deserves context.</h3>
                    <p>
                      Use your insights as a starting point for a conversation
                      with your care provider.
                    </p>
                    <button className="text-link" onClick={() => setHelp(true)}>
                      How this workspace works
                    </button>
                  </section>
                </div>
              </>
            )}
            {page === "profile" && (
              <Profile
                values={values}
                meta={meta}
                setMeta={updateMeta}
                setValue={updateValue}
                loadSample={example}
                importText={importText}
                importDocument={importDocument}
                run={screen}
                busy={!!busy}
                sample={sample}
                reviewed={reviewed}
                setReviewed={setReviewed}
              />
            )}
            {page === "insights" &&
              (screening ? (
                <>
                  <div className="note">
                    <ShieldCheck size={19} />
                    These are research estimates, not a diagnosis. The model was
                    evaluated on US data; prediction accuracy in Indian patients
                    has not been established.
                  </div>
                  <div className="risk-grid">
                    {(
                      [
                        {
                          key: "diabetes",
                          label: "Diabetes",
                          icon: HeartPulse,
                        },
                        {
                          key: "obesity",
                          label: "Weight-related health",
                          icon: UserRound,
                        },
                        { key: "cvd", label: "Heart health", icon: HeartPulse },
                      ] as const
                    ).map((r) => (
                      <article className="card risk-card" key={r.key}>
                        <span className="icon-tile">
                          <r.icon size={22} />
                        </span>
                        <h3>{r.label}</h3>
                        <div className="risk-number">
                          {(screening.probabilities[r.key] * 100).toFixed(1)}
                          <span>%</span>
                        </div>
                        <span
                          className={`risk-label ${screening.classifications[r.key].positive ? "attention" : ""}`}
                        >
                          {screening.classifications[r.key].positive
                            ? "Discuss with your care provider"
                            : "Below research threshold"}
                        </span>
                        <details>
                          <summary>What does this mean?</summary>
                          <p>
                            This is the estimated likelihood returned for this
                            profile. The research threshold is{" "}
                            {(
                              screening.classifications[r.key].threshold * 100
                            ).toFixed(1)}
                            %. This is not a clinically validated treatment
                            threshold and cannot establish or rule out a condition.
                          </p>
                        </details>
                      </article>
                    ))}
                  </div>
                  <section className="card energy-summary">
                    <div>
                      <span className="eyebrow">
                        YOUR NUTRITION STARTING POINT
                      </span>
                      <h2>A daily target to plan around.</h2>
                      <p className="muted">
                        A starting point based on your profile. Your needs may
                        change over time.
                      </p>
                    </div>
                    <div>
                      <Flame size={22} />
                      <strong>
                        {Math.round(
                          screening.energy_prescription.target_calories,
                        ).toLocaleString()}
                      </strong>
                      <span>kcal / day</span>
                    </div>
                  </section>
                  {!screening.certified_for_stage3 && (
                    <div className="alert">
                      Your assessment needs review before a meal plan can be
                      created. Please review the profile with your care
                      provider.
                    </div>
                  )}
                  <div className="page-actions">
                    <Button variant="secondary" onClick={() => go("profile")}>
                      Review my profile
                    </Button>
                    <Button
                      disabled={!screening.certified_for_stage3}
                      onClick={() => go("preferences")}
                    >
                      Choose my food preferences
                    </Button>
                  </div>
                </>
              ) : (
                <Empty
                  icon={HeartPulse}
                  title="Let’s start with your profile."
                  action={
                    <Button onClick={() => go("profile")}>
                      Complete my profile
                    </Button>
                  }
                >
                  Your health insights will appear after you review your details
                  and complete an assessment.
                </Empty>
              ))}
            {page === "preferences" && (
              <div className="preferences-layout">
                <section className="card">
                  <div className="card-title">
                    <div>
                      <span className="eyebrow">
                        A PLAN THAT FITS YOUR LIFE
                      </span>
                      <h2>What feels right on your plate?</h2>
                    </div>
                    <SlidersHorizontal size={24} />
                  </div>
                  <p className="muted">
                    Choose what you enjoy and tell us what needs to stay off the
                    menu.
                  </p>
                  <fieldset disabled={!!busy}>
                    <div className="preferences-fields">
                      <Select
                        label="How do you prefer to eat?"
                        value={prefs.dietary_preference}
                        onChange={(e) => {
                          changedPrefs();
                          setPrefs({
                            ...prefs,
                            dietary_preference: e.target
                              .value as MealPlanPreferences["dietary_preference"],
                          });
                        }}
                      >
                        <option value="pure_vegetarian">Vegetarian</option>
                        <option value="veg_nonveg_combo">
                          Vegetarian & non-vegetarian
                        </option>
                        <option value="pure_non_vegetarian">
                          Non-vegetarian
                        </option>
                      </Select>
                      <Select
                        label="How much variety would you like?"
                        value={prefs.max_active_dishes ?? ""}
                        onChange={(e) => {
                          changedPrefs();
                          setPrefs({
                            ...prefs,
                            max_active_dishes: e.target.value
                              ? Number(e.target.value)
                              : null,
                          });
                        }}
                      >
                        <option value="">Choose for me</option>
                        <option value="4">
                          Keep it simple · up to 4 dishes
                        </option>
                        <option value="5">
                          A little variety · up to 5 dishes
                        </option>
                        <option value="6">
                          Balanced variety · up to 6 dishes
                        </option>
                        <option value="7">More variety · up to 7 dishes</option>
                        <option value="8">
                          Plenty of choice · up to 8 dishes
                        </option>
                      </Select>
                      <Field
                        label="Allergies & intolerances"
                        placeholder="e.g. peanuts, milk, shellfish"
                        value={allergies}
                        maxLength={500}
                        onChange={(e) => {
                          changedPrefs();
                          setAllergies(e.target.value);
                        }}
                        hint="Separate each item with a comma. Leave blank only if you have none."
                      />
                      <Field
                        label="Foods you prefer to avoid"
                        placeholder="e.g. cabbage, okra"
                        value={dislikes}
                        maxLength={500}
                        onChange={(e) => {
                          changedPrefs();
                          setDislikes(e.target.value);
                        }}
                        hint="Separate dishes or ingredients with commas."
                      />
                      <details className="preference-help">
                        <summary>How are my preferences used?</summary>
                        <p>
                          Your preferences are sent with your current assessment
                          when you create a plan. Changing this form does not
                          alter your existing plan until you create a new one.
                          Allergy data may be incomplete; review the returned
                          plan and any warnings with your care provider.
                        </p>
                      </details>
                      <label className="check-label">
                        <input
                          type="checkbox"
                          checked={prefReview}
                          onChange={(e) => setPrefReview(e.target.checked)}
                        />
                        <span>
                          I have reviewed my diet choice, allergies, and foods
                          to avoid.
                        </span>
                      </label>
                    </div>
                    <div className="button-row">
                      <Button
                        disabled={
                          !screening?.certified_for_stage3 || !prefReview
                        }
                        onClick={generate}
                      >
                        {plan?.accepted
                          ? "Create updated meal plan"
                          : "Create my meal plan"}
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => {
                          setPrefs({ ...initialPrefs });
                          setAllergies("");
                          setDislikes("");
                          setPrefReview(false);
                          setToast(
                            "Preferences reset. Review them before creating a plan.",
                          );
                        }}
                      >
                        Reset preferences
                      </Button>
                    </div>
                  </fieldset>
                </section>
                <aside className="preferences-side">
                  <div className="card subtle">
                    <Leaf size={25} />
                    <h3>
                      Familiar food.
                      <br />
                      Thoughtfully chosen.
                    </h3>
                    <p>
                      We’ll use these choices to build a daily plan around the
                      foods you can enjoy.
                    </p>
                  </div>
                  <div className="card">
                    <h3>
                      {screening
                        ? "Your profile is ready."
                        : "Complete your profile first."}
                    </h3>
                    <p className="muted">
                      {screening
                        ? "You can review your health insights before creating your plan."
                        : "You can choose preferences now. A health assessment is needed to generate your meals."}
                    </p>
                    <Button
                      variant="secondary"
                      onClick={() => go(screening ? "insights" : "profile")}
                    >
                      {screening ? "View health insights" : "Go to my profile"}
                    </Button>
                  </div>
                </aside>
              </div>
            )}
            {page === "meals" &&
              (plan?.accepted ? (
                <MealPlan
                  key={revision.current}
                  plan={plan}
                  onEdit={() => go("preferences")}
                  onChat={() => go("assistant")}
                />
              ) : (
                <Empty
                  icon={Utensils}
                  title={
                    plan
                      ? "Let’s adjust your plan."
                      : "Your next meal starts here."
                  }
                  action={
                    <Button
                      onClick={() => go(screening ? "preferences" : "profile")}
                    >
                      {screening
                        ? "Review preferences"
                        : "Start with my profile"}
                    </Button>
                  }
                >
                  {plan
                    ? "We couldn’t create a plan that met the required nutrition checks. Review your preferences with your care provider and try again."
                    : "Complete your profile and choose your food preferences to create a personal daily menu."}
                </Empty>
              ))}
            {page === "assistant" &&
              (plan?.accepted ? (
                <div className="assistant-layout">
                  <section className="card chat-panel">
                    <div className="card-title">
                      <div className="assistant-identity">
                        <span className="icon-tile">
                          <MessageCircle size={23} />
                        </span>
                        <div>
                          <h2>Your plan assistant</h2>
                          <p className="muted">
                            A little help with your everyday choices.
                          </p>
                        </div>
                      </div>
                    </div>
                    {flagged && (
                      <div className="alert">
                        Your plan needs review for a reported allergy or
                        intolerance. Please review it before use.
                      </div>
                    )}
                    <div className="chat-messages" aria-live="polite">
                      {!turns.length && (
                        <div className="chat-welcome">
                          <span className="icon-tile">
                            <Leaf size={30} />
                          </span>
                          <h3>Let’s make your plan work for you.</h3>
                          <p>
                            Ask about a portion, explore a food swap, or
                            understand what’s in your plan.
                          </p>
                          <div className="suggestion-grid">
                            {[
                              "Explain my meal portions",
                              "How much protein and fibre is in lunch?",
                              "Can I replace rice in my plan?",
                            ].map((q) => (
                              <button
                                disabled={!!busy}
                                key={q}
                                onClick={() => send(q)}
                              >
                                {q}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {turns.map((t, i) => (
                        <div key={i} className="chat-turn">
                          <div className="user-message">{t.question}</div>
                          <div className="assistant-message">
                            <span className="message-label">
                              <Leaf size={14} />
                              YOUR ASSISTANT
                            </span>
                            <p>{t.response.answer}</p>
                            {t.response.interpreted_request && (
                              <small>Understood as: {t.response.interpreted_request}</small>
                            )}
                            {["unavailable", "busy", "rejected", "semantic_rejected"].includes(t.response.interpretation_status || "") && (
                              <small>Natural-language interpretation was unavailable or unclear. Please clarify your request.</small>
                            )}
                            {!!t.response.clarification_options?.length && (
                              <div className="suggestion-grid">
                                {t.response.clarification_options.map((option) => (
                                  <button key={option} disabled={!!busy || i !== turns.length - 1}
                                    onClick={() => send(option)}>{option}</button>
                                ))}
                              </div>
                            )}
                            {t.response.proposed_meal_plan && (
                              <div className="proposal">
                                <h4>Suggested plan update</h4>
                                <p>
                                  Review the proposed food quantities before
                                  applying this change.
                                </p>
                                <details>
                                  <summary>View proposed foods</summary>
                                  <ul>
                                    {t.response.proposed_meal_plan.solved_food_basket.map(
                                      (f, j) => (
                                        <li key={j}>
                                          {f.name} · {f.quantity}{" "}
                                          {f.serving_unit}
                                        </li>
                                      ),
                                    )}
                                  </ul>
                                </details>
                                <Button
                                  disabled={
                                    !!busy ||
                                    t.applied ||
                                    t.revision !== revision.current
                                  }
                                  onClick={() => accept(i)}
                                >
                                  {t.applied
                                    ? "Applied to my plan"
                                    : t.revision !== revision.current
                                      ? "Replaced by a newer plan"
                                      : "Apply this change"}
                                </Button>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      <div ref={endOfChat} />
                    </div>
                    <form
                      className="chat-composer"
                      onSubmit={(e) => {
                        e.preventDefault();
                        send(question);
                      }}
                    >
                      <label className="sr-only" htmlFor="chat-input">
                        Message your plan assistant
                      </label>
                      <textarea
                        id="chat-input"
                        placeholder="Ask about your plan or request a change…"
                        rows={2}
                        maxLength={1000}
                        value={question}
                        disabled={!!busy}
                        onChange={(e) => setQuestion(e.target.value)}
                      />
                      <button
                        className="send-button"
                        aria-label="Send message"
                        disabled={!!busy || !question.trim()}
                      >
                        <Send size={19} />
                      </button>
                    </form>
                    <label>
                      <input type="checkbox" checked={naturalChat} disabled={!!busy}
                        onChange={(e) => setNaturalChat(e.target.checked)} />
                      Understand natural wording (experimental; responses may take longer)
                    </label>
                    <p className="chat-footnote">
                      Proposed changes are only applied when you accept them.
                    </p>
                  </section>
                  <aside className="assistant-side">
                    <section className="card">
                      <History size={23} />
                      <h3>A little room to change your mind.</h3>
                      <p>
                        Review saved plan versions and restore an earlier one.
                      </p>
                      <Button
                        variant="secondary"
                        disabled={!!busy}
                        onClick={loadHistory}
                      >
                        <History size={16} />
                        View plan history
                      </Button>
                      {showHistory && history && (
                        <div className="history-list">
                          {history.versions.map((v) => (
                            <div key={v.version}>
                              <div>
                                <strong>Version {v.version}</strong>
                                <small>
                                  {new Date(v.created_at).toLocaleString()}
                                </small>
                              </div>
                              <Button
                                variant="ghost"
                                disabled={
                                  !!busy ||
                                  v.version === history.current_version
                                }
                                onClick={() => restore(v.version)}
                              >
                                {v.version === history.current_version
                                  ? "Current"
                                  : "Restore"}
                              </Button>
                            </div>
                          ))}
                        </div>
                      )}
                    </section>
                    <section className="card subtle">
                      <h3>Ready to see your changes?</h3>
                      <p>
                        Your meal plan always reflects the latest accepted
                        version.
                      </p>
                      <Button variant="secondary" onClick={() => go("meals")}>
                        Open my meal plan
                      </Button>
                    </section>
                  </aside>
                </div>
              ) : (
                <Empty
                  icon={MessageCircle}
                  title="Your assistant needs a plan to start."
                  action={
                    <Button
                      onClick={() => go(screening ? "preferences" : "profile")}
                    >
                      Create my plan
                    </Button>
                  }
                >
                  Once your meal plan is ready, ask questions and request
                  changes here.
                </Empty>
              ))}
            {page === "reports" &&
              (report ? (
                <Report report={report} />
              ) : (
                <Empty
                  icon={FileText}
                  title={
                    plan?.accepted
                      ? "Bring your insights together."
                      : "Your nutrition story starts here."
                  }
                  action={
                    <Button
                      disabled={!!busy}
                      onClick={() =>
                        plan?.accepted
                          ? makeReport()
                          : go(screening ? "preferences" : "profile")
                      }
                    >
                      {plan?.accepted ? "Create my report" : "Create my plan"}
                    </Button>
                  }
                >
                  {plan?.accepted
                    ? "Create a clear summary of your health insights, meals, and everyday guidance. Download it or print it for your next appointment."
                    : "Your personal report will become available after your meal plan is ready."}
                </Empty>
              ))}
          </div>
          <footer>
            <span>
              <Leaf size={14} />
              Thoughtful nutrition. Everyday living.
            </span>
            <span>Personal guidance · Review with your care provider</span>
          </footer>
        </main>
      </div>
      {toast && (
        <div className="toast" role="status">
          <CheckCircle2 size={19} />
          {toast}
          <button
            aria-label="Dismiss notification"
            onClick={() => setToast("")}
          >
            <X size={16} />
          </button>
        </div>
      )}
      <dialog
        ref={dialog}
        className="help-dialog"
        onCancel={() => setHelp(false)}
        onClick={(e) => {
          if (e.target === dialog.current) setHelp(false);
        }}
      >
        <div className="dialog-heading">
          <span className="icon-tile">
            <HelpCircle size={24} />
          </span>
          <button
            className="icon-button"
            aria-label="Close help"
            onClick={() => setHelp(false)}
          >
            <X />
          </button>
        </div>
        <h2>A simpler way to get started.</h2>
        <ol>
          <li>
            <strong>Complete your profile.</strong> Import a prepared TXT file
            or fill the grouped sections.
          </li>
          <li>
            <strong>Review your insights.</strong> Then choose your diet,
            allergies, and preferences.
          </li>
          <li>
            <strong>Create your meal plan.</strong> Ask for changes, download a
            shopping list, or create a report.
          </li>
        </ol>
        <div className="connection-box">
          <strong>
            {connected === null
              ? "Checking service…"
              : connected
                ? "Nutrition service connected"
                : "Nutrition service is unavailable"}
          </strong>
          <p>
            {connected
              ? "You can run assessments and create plans."
              : "Start the supplied nutrition service to use assessments, meal generation and the assistant. Your form remains available."}
          </p>
          <Button
            variant="secondary"
            disabled={!!busy}
            onClick={() =>
              act("Checking connection…", async () => {
                try {
                  const h = await api.health();
                  setConnected(h.pipeline_loaded);
                  setToast(
                    h.pipeline_loaded
                      ? "Connection is ready."
                      : "The service is still preparing.",
                  );
                } catch {
                  setConnected(false);
                  setToast("The service is still unavailable.");
                }
              })
            }
          >
            <RefreshCw size={16} />
            Check connection
          </Button>
        </div>
        <p className="help">
          Health data remains in this page’s memory. Reloading clears the
          current workspace. Accepted plan history is saved by the connected
          service. Water entries are saved locally on this device.
        </p>
        <Button onClick={() => setHelp(false)}>Got it</Button>
      </dialog>
    </div>
  );
}
function FlaskIcon() {
  return (
    <span className="sample-symbol" aria-hidden="true">
      i
    </span>
  );
}
