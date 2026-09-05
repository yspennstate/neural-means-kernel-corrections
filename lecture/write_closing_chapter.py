"""Write the closing chapter only from the complete, checked campaign figures."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def segment(line,math,say):return dict(line=line,math=math,say=say)


def board(key,title,kind,label,argument,segments,**visual):
    return dict(key=key,title=title,visual=dict(kind=kind,label=label,
        source='paper/theory.tex and the completed sensitivity evidence',**visual),
        visual_argument=argument,segments=segments)


def main():
    path=HERE/'assets/sensitivity_figures.json'
    record=json.loads(path.read_text(encoding='utf-8'));data=record['summary']
    for row in record['rows']:
        if hashlib.sha256((HERE/'assets'/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Closing figure identity failed')
    if data['independent_prediction_checks']['scenarios']!=18:
        raise ValueError('Closing chapter requires the entire verified campaign')
    cent=data['centering']['aggregate'];mis=data['mismatch']['aggregate']
    delta=100*cent['local_minus_pooled']['mean']
    mismatch_delta=100*mis['test_consistent_minus_historical']['mean']
    direction='higher' if delta>0 else 'lower' if delta<0 else 'unchanged'
    mismatch_direction='increases' if mismatch_delta>0 else 'decreases' if mismatch_delta<0 else 'leaves unchanged'
    def pp(value):
        mantissa,exponent=f'{value:.2e}'.split('e')
        return mantissa+rf'\times10^{{{int(exponent)}}}'
    boards=[board('c12_layers','Read the pipeline one layer at a time','pipeline_stages',
        'Diagram of the mechanics pipeline; each fitted object has a different statistical role',
        'The arrows separate construction of a mean, correction of its remaining error, and calibration of a radius.',[
        segment('First choose a useful mean',r'm(u)=\operatorname{Stack}(f_1(u),\ldots,f_M(u))',
            'The first question is whether the available predictors make different mistakes. Averaging reduces the part of their error that points in different directions. Repeating a configuration can reduce variation among its seeds, but a shared component remains. The block calculation quantifies this obstruction under its stated assumptions. It does not say that every extra model must be useless.'),
        segment('Then approximate what the mean leaves behind',r'r(u)=G(u)-m(u)',
            'The next question concerns the remaining function. A kernel correction uses its observed values and the geometry of the kernel sections. The sharp error factor measures the part of a query section that the chosen reconstruction misses. This is a worst-case statement over a specified function-space ball. Its radius must be supplied; the data do not reveal it automatically.'),
        segment('Calibrate a prediction set on separate cases',r'C(u)=\{v:\|v-\widehat G(u)\|_2\le Qa(u)\}',
            'The third question is statistical coverage. The rank argument calibrates a ball around a fixed prediction, with any positive scale fixed in advance. It uses exchangeability instead of an unknown native-space norm. These three questions explain why ensemble accuracy, kernel geometry, and calibration appear together in the paper. They are connected, but each answer has its own assumptions.')]),
    board('c12_centering','Measure the implementation differences','sensitivity_results',
        'Completed ten-seed comparisons; percentage-point differences; the two vertical scales are separate',
        'Paired differences reveal changes hidden by nearly overlapping headline accuracy values.',[
        segment('Pair each fold-local run with its pooled counterpart',r'\Delta_s=100(E_{s,\mathrm{local}}-E_{s,\mathrm{pooled}})',
            'The top panel compares the two centering conventions within each training seed. Every dot uses the same benchmark cases and a paired downstream rerun. Read the axis multiplier before judging the height of a line. Pairing isolates this implementation difference more clearly than comparing two unrelated runs, whose changes could also come from initialization or data ordering.'),
        segment('Report the size as well as the sign',rf'\overline\Delta={pp(delta)}\ \mathrm{{pp}}',
            f'Across the ten pairs, the fold-local mean error is {direction}. The average change has magnitude about {abs(delta)*1e6:.2f} millionths of a percentage point. The number is deliberately shown on its own scale. Its magnitude supports a limited statement about these retained predictors and cases. The validity of pooled target centering as cross-fitting is a separate question about which labels each fit may use.'),
        segment('The correction-label check is a separate experiment',rf'\overline\Delta_{{\mathrm{{labels}}}}={pp(mismatch_delta)}\ \mathrm{{pp}}',
            f'The lower panel holds the historical mean, kernel, and stacking weights fixed, and changes only the residual labels to match the deployed mean. This {mismatch_direction} the mean test error by about {abs(mismatch_delta)*1e6:.2f} millionths of a percentage point. The propagated mismatch term also bounds the casewise change by the reverse triangle inequality. A bound on that change is not a prediction of its sign.')],chart='centering')]
    for metric,readable in (('reduced','reduced-coordinate'),('radiance','reconstructed-radiance')):
        improved=sum(v<0 for band in data['oco_grids'].values()
                     for v in band['differences']['combined_'+metric]['per_seed'])
        boards.append(board('c12_grid_'+metric,
            'A larger kernel search: '+readable+' error','sensitivity_results',
            'Completed three-seed comparisons in each band; negative values mean lower error under the expanded grid',
            'The paired error differences show which fitted families benefit from the larger validation search.',[
            segment('Change the search while retaining the other controls',r'4\times3=12\quad\longrightarrow\quad7\times8=56',
                (f'This panel uses {readable} error. The grid grows from twelve to fifty-six candidates by extending the kernel scales and nuggets. Each comparison retains its network features, split, and source emulator. The dots represent the three paired seeds within each spectral band. The short horizontal marks show their means, not confidence intervals for new atmospheric states.' if metric=='reduced' else
                 'Now keep exactly the same predictions and examine reconstructed radiance. Undo the coordinate standardization, apply the spectral basis, and restore the reference mean. This changes the norm in which an error is measured. A unit error in one reduced coordinate can affect the spectrum much more than the same error in another. The signs and relative sizes of improvements can therefore change.')),
            segment('Read all three fitted families',r'\Delta=100(E_{\mathrm{expanded}}-E_{\mathrm{recorded}})',
                ('The raw-input kernel tests how much the original comparison depended on a restricted kernel search. The feature head tests the same issue after a neural representation has been learned. The coordinate combination is selected again from validation errors, because its available candidates have changed. Looking at all three families prevents an improvement in one component from being mistaken for an improvement everywhere.' if metric=='reduced' else
                 'Compare each spectral band with its reduced-coordinate panel. The plotted changes are derived from the same saved predictions, rather than a separately selected set of favorable runs. We do not choose another model after looking at this test panel. The radiance calculation was also checked through a spectral Gram quadratic form and by direct summation across the reconstructed channels.')),
            segment('Validation selection does not force test improvement',rf'\#\{{\Delta_{{\mathrm{{combination}}}}<0\}}={improved}\ \mathrm{{of}}\ 9',
                (f'The coordinate combination improves in {improved} of the nine paired seed-and-band comparisons in this metric. That count describes these comparisons; it is not a significance test with nine independent datasets. A larger candidate set cannot worsen the best validation objective when the old candidates are retained, but it can change selection in a way that worsens test error.' if metric=='reduced' else
                 f'In reconstructed radiance, the coordinate combination improves in {improved} of the nine paired comparisons. The practical interpretation depends on the prediction task: reduced-coordinate accuracy and spectral accuracy answer different questions. Report both when both matter. A method should earn its conclusion in the declared metric, under a stated search budget, with the difficult cases still visible.'))],chart='grid_'+metric))
    boards.extend([
    board('c12_scope','Keep the theorem tied to its assumptions','kernel_projection',
        'Exact two-feature illustration: training section (1,0), query section (3,2), interpolation coefficient 3',
        'The unobserved component explains both the sharp bound and why an unknown norm remains necessary.',[
        segment('The sharp factor is a norm of a residual representer',r'\sup_{\|r\|_K\le\rho}\|r(u)-\widehat r_\lambda(u)\|_2=\rho\widetilde P_\lambda(u)',
            'Return to the projection picture. The residual representer gives the exact worst-case constant for the fixed kernel reconstruction. Cauchy and Schwarz give the upper bound; a residual aligned with that representer attains it. Both parts are necessary for sharpness. The finite-dimensional drawing makes this alignment visible, while the Hilbert-space proof supplies the general statement.'),
        segment('Indistinguishable residuals give the minimax lower bound',r'r^+(X)=r^-(X)=0',
            'For interpolation, the residual representer vanishes at the training inputs. Two opposite multiples therefore produce identical observations and opposite query values. Every estimator returns the same answer for both, so one error is at least half their separation. This is the main information argument. It applies to rules using those observations, with the kernel and function class fixed.'),
        segment('The reported geometry does not estimate the missing norm',r'\rho\ \text{must be justified separately}',
            'The calculation gives a geometric constant, not an estimate of the unknown target norm. Likewise, a small fitted correction norm does not prove that the true residual is smooth or small in that space. The paper therefore reports measured errors and calibrated radii alongside the deterministic theory. Their agreement or disagreement is informative precisely because the claims are kept distinct.')]),
    board('c12_return','Return to the prediction problem','elastic_domain',
        'The mechanics task maps a boundary load to a field; accuracy and uncertainty concern that whole output',
        'Returning to the physical input-output map connects the abstract argument to the task being solved.',[
        segment('Check the prediction in the metric that matters',r'E=\frac1N\sum_i\frac{\|\widehat G(u_i)-G(u_i)\|_2}{\|G(u_i)\|_2}',
            'For a new application, start by writing the error metric and the intended output. Check ordinary cases and difficult cases in physical coordinates. A mean percentage alone will not show whether an error changes a localized stress peak or a narrow spectral feature. The field and spectrum images in this lecture are part of the numerical argument, rather than decoration.'),
        segment('Use new cases to test the completed choices',r'\text{fit}\ \longrightarrow\ \text{calibrate}\ \longrightarrow\ \text{evaluate}',
            'Freeze the predictor and every tuning choice before using fresh calibration and evaluation cases. The current benchmarks have been inspected through many research decisions, so another split of the same archive does not erase that history. A fresh campaign would test whether the observed gains and coverage survive a new collection of cases and any intended change of operating conditions.'),
        segment('Three objects make the method understandable',r'\text{error vectors},\quad\text{kernel sections},\quad\text{score ranks}',
            'Keep three pictures in mind: error vectors reveal what averaging can cancel; kernel sections reveal what observations can reconstruct; and score ranks explain marginal coverage. They provide different ways to question a surrogate. Together with a reproducible experiment and images of its successes and failures, they make the method something we can inspect, prove statements about, and test.')])])
    result=dict(id='12',title='What the proofs and completed experiments establish',boards=boards,
                evidence_manifest_sha256=record['evidence_manifest_sha256'])
    output=HERE/'chapters/12.json'
    if output.exists():raise ValueError('Refuse to overwrite an existing closing chapter')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Closing chapter written:',len(boards),'boards;',sum(len(b['segments']) for b in boards),'segments')


if __name__=='__main__':main()
