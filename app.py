"""Streamlit assessment analytics and prediction dashboard."""
from pathlib import Path
import io
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from insights import validate_records, summarize, comparison, action

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Mathematics Assessment ABI', page_icon='📘', layout='wide')
st.title('Intelligent Mathematics Assessment Analytics')
st.caption('Understand past performance • estimate next-attempt correctness • refresh teacher priorities')

@st.cache_data
def load():
    return pd.read_csv(ROOT/'data'/'assessment_features.csv'), json.loads((ROOT/'outputs'/'metrics.json').read_text())

df, info = load()
skills = sorted(df.skill_name.dropna().unique())
selected = st.sidebar.multiselect('Filter mathematical skills', skills)
def filtered(data):
    return data[data.skill_name.isin(selected)] if selected else data
view = filtered(df)
st.sidebar.caption('Filters apply to historical analytics and adaptive insights. Prediction uses the skill selected in its own tab.')
tab1, tab2, tab3 = st.tabs(['📊 Assessment analytics', '🧠 Predict correctness', '🔁 Adaptive insights'])

with tab1:
    a, b, c = st.columns(3)
    a.metric('Question attempts', f'{len(view):,}')
    b.metric('Observed correct rate', f'{view.correct.mean():.1%}' if len(view) else '—')
    c.metric('Students represented', f'{view.user_id.nunique():,}')
    st.caption('Each record is one question interaction; correctness is the dataset’s first-attempt outcome. Counts are records, not repeat tries within a question.')
    stats = summarize(view).reset_index()
    lowest = stats[stats.attempts >= 30].sort_values('correct_rate').head(15)
    if not lowest.empty:
        fig = px.bar(lowest, x='correct_rate', y='skill_name', orientation='h', custom_data=['attempts'], title='Lowest historical correctness • at least 30 attempts', labels={'correct_rate':'Correct rate', 'skill_name':'Mathematics skill'})
        fig.update_xaxes(tickformat='.0%', range=[0, 1])
        fig.update_traces(hovertemplate='%{y}<br>Correct: %{x:.1%}<br>Attempts: %{customdata[0]:,}<extra></extra>')
        fig.update_layout(height=530, yaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    st.subheader('Skill performance and evidence')
    st.dataframe(stats.style.format({'correct_rate':'{:.1%}', 'attempts':'{:,.0f}'}), hide_index=True)
    st.subheader('Sample of historical records')
    st.dataframe(view[['user_id','problem_id','skill_name','correct']].head(100), hide_index=True)
    st.caption('Outcome key: 1 = correct; 0 = incorrect.')

with tab2:
    st.subheader('Predict a future question attempt')
    st.info('Use only information known before the question. Choose a consistent student history; these controls demonstrate the saved model and do not look up an individual student.')
    model_choice = st.selectbox('Model', ['Logistic Regression','Decision Tree'])
    skill = st.selectbox('Mathematical skill', skills, key='p_skill')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('### 1. Student across all skills')
        st.caption('The student’s total previous question interactions and percentage answered correctly.')
        student_n = st.number_input('Student previous attempts (all skills)', min_value=0, value=10)
        student_pct = st.slider('Student previous correct rate (%)', 0, 100, 60, 5)
    with c2:
        st.markdown('### 2. Student in this skill')
        st.caption('The subset of that student’s history for the selected mathematics skill.')
        both_n = st.number_input('Student previous attempts in this skill', min_value=0, value=3)
        both_pct = st.slider('Student previous correct rate in this skill (%)', 0, 100, 50, 5)
    with c3:
        st.markdown('### 3. All students in this skill')
        st.caption('Earlier interactions across students. Defaults illustrate this skill’s historical context; adjust to the intended prediction time.')
        skill_n = st.number_input('Earlier observed attempts in this skill', min_value=0, value=int(min((df.skill_name == skill).sum()//2, 1000)))
        skill_pct = st.slider('Earlier observed correct rate for this skill (%)', 0, 100, int(round(df.loc[df.skill_name == skill, 'correct'].mean()*100)))
    st.caption('With zero previous attempts, the existing model convention uses a neutral 50% prior rate, regardless of the slider. Small histories provide limited evidence.')
    if st.button('Predict correctness probability', type='primary'):
        if both_n > student_n or both_n > skill_n:
            st.error('Student attempts in this skill cannot exceed either all student attempts or all observed attempts in the skill.')
        else:
            model = joblib.load(ROOT/'models'/('logistic_regression.joblib' if model_choice == 'Logistic Regression' else 'decision_tree.joblib'))
            row = pd.DataFrame([{'student_prior_n':student_n, 'student_prior_rate':student_pct/100 if student_n else .5, 'skill_prior_n':skill_n, 'skill_prior_rate':skill_pct/100 if skill_n else .5, 'student_skill_prior_n':both_n, 'student_skill_prior_rate':both_pct/100 if both_n else .5, 'skill_name':skill}])
            p = float(model.predict_proba(row)[0, 1])
            a, b = st.columns(2)
            a.metric('Estimated chance of a correct first attempt', f'{p:.1%}')
            b.metric('Estimated chance of an incorrect first attempt', f'{1-p:.1%}')
            st.progress(p)
            st.write(f'The model estimates about {round(p*100)} correct first attempts per 100 comparable attempts. This is a probability estimate, not a grade or certainty.')
            st.success('Suggested teacher action: offer guided practice and check misconceptions.' if p < .6 else 'Suggested teacher action: continue practice and check understanding with a follow-up question.')
            st.caption('The 60% action threshold is an illustrative teaching rule, not a clinically or educationally validated cutoff. Predictions do not diagnose learning difficulties.')
    st.subheader('Held-out test performance')
    scores = pd.DataFrame({k:{m:v for m,v in val.items() if m != 'confusion_matrix'} for k,val in info['metrics'].items()}).T
    st.dataframe(scores.style.format({k: ('{:.1%}' if k in ['accuracy','precision','recall','f1'] else '{:.3f}') for k in scores.columns}))
    st.caption('Accuracy, precision, recall and F1 are displayed as percentages. ROC-AUC is a discrimination score (higher is better); Brier is probability error (lower is better). Evaluation uses the final 20% of chronological interactions.')

with tab3:
    st.subheader('See how new evidence changes teacher priorities')
    st.write('Compare historical records with a temporary CSV update. Uploads refresh the analysis below; they do not save to disk or retrain either prediction model.')
    minimum = st.slider('Minimum attempts per skill to make a recommendation', 10, 200, 50, 10)
    threshold_pct = st.slider('Flag a skill below this correct rate (%)', 20, 90, 60, 5)
    threshold = threshold_pct / 100
    st.caption('Below the minimum: collect evidence. At or above the minimum and below the rate threshold: review misconceptions and offer guided practice. Otherwise: continue practice and monitor.')
    source = st.radio('Update source', ['No update', 'Built-in demonstration', 'Upload CSV'], horizontal=True)
    new = pd.DataFrame(columns=['user_id','problem_id','skill_name','correct'])
    if source == 'Built-in demonstration':
        # Pick a flagged skill and add enough successes to cross the current threshold.
        candidates = summarize(view)
        candidates = candidates[(candidates.attempts >= minimum) & (candidates.correct_rate < threshold)]
        if candidates.empty:
            st.info('No eligible flagged skill for these filters and thresholds. Change the controls or upload your own CSV.')
        else:
            target = candidates.sort_values('attempts').index[0]
            n, rate = candidates.loc[target, ['attempts','correct_rate']]
            count = int((threshold*n - rate*n)/(1-threshold)) + 2
            new = pd.DataFrame({'user_id':['DEMO_STUDENT']*count, 'problem_id':[f'DEMO_{i}' for i in range(count)], 'skill_name':[target]*count, 'correct':[1]*count})
            st.info(f'Synthetic demonstration: {count:,} correct attempts for “{target}”. This deliberately crosses the selected threshold to illustrate the rule; it is not observed learning progress.')
            st.download_button('Download demonstration CSV', new.to_csv(index=False).encode(), 'demo_attempts.csv', 'text/csv')
    elif source == 'Upload CSV':
        st.caption('Required columns: user_id, problem_id, skill_name, correct (0 or 1). Every row is treated as a new interaction; repeated uploads replace the preview and never accumulate. Ensure records do not overlap the historical dataset.')
        st.download_button('Download CSV template', b'user_id,problem_id,skill_name,correct\n', 'attempts_template.csv', 'text/csv')
        upload = st.file_uploader('New assessment attempt records', type=['csv'])
        if upload is not None:
            try:
                new, rejected = validate_records(pd.read_csv(io.BytesIO(upload.getvalue())))
                st.success(f'Accepted {len(new):,} records; rejected {rejected:,} invalid records.')
                if rejected:
                    st.warning('Rejected rows have blank identifiers/skills or a correctness value other than 0 or 1.')
            except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
                st.error(f'CSV could not be applied: {exc}')
    scoped_new = filtered(new)
    combined = pd.concat([view, scoped_new], ignore_index=True)
    changes = comparison(view, combined, minimum, threshold)
    flagged_before = ((changes.attempts_before >= minimum) & (changes.correct_rate_before < threshold)).sum()
    flagged_after = ((changes.attempts_after >= minimum) & (changes.correct_rate_after < threshold)).sum()
    a,b,c = st.columns(3)
    a.metric('New attempts in current filter', f'{len(scoped_new):,}')
    b.metric('Skills flagged after update', int(flagged_after), int(flagged_after-flagged_before), delta_color='inverse')
    c.metric('Recommendations changed', int(changes.recommendation_changed.sum()))
    st.caption(f'Before: {len(view):,} attempts and {flagged_before:,} flagged skills. After: {len(combined):,} attempts and {flagged_after:,} flagged skills. {len(new)-len(scoped_new):,} new rows are outside the current skill filter.')
    st.subheader('Before / after impact by skill')
    affected = changes[changes.added_attempts > 0]
    if affected.empty:
        st.info('Choose the demonstration or upload valid in-scope records to see before/after changes.')
    else:
        st.dataframe(affected.style.format({'correct_rate_before':'{:.1%}', 'correct_rate_after':'{:.1%}', 'change_pp':'{:+.2f}'}, na_rep='—'), hide_index=True)
    st.caption('change_pp means percentage points (for example, 50% → 60% is +10 points). A new skill has no historical rate, shown as —.')
    st.subheader('Current teacher recommendations')
    current = summarize(combined).reset_index()
    current['recommendation'] = [action(n,r,minimum,threshold) for n,r in zip(current.attempts,current.correct_rate)]
    current = current.sort_values(['correct_rate','attempts'], ascending=[True,False])
    st.dataframe(current.style.format({'correct_rate':'{:.1%}', 'attempts':'{:,.0f}'}), hide_index=True)
    st.download_button('Download before / after report', changes.to_csv(index=False).encode(), 'adaptive_impact.csv', 'text/csv')
    st.caption('Recommendations summarize group evidence and support teacher judgement. Low correctness alone does not establish a learning disability or a causal teaching issue. Saved models and their held-out metrics remain unchanged.')
