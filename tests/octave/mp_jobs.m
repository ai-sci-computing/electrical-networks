function mp_jobs(mproot, jobfile)
  % Runs MATPOWER 8.1 jobs for tests/matpower_octave.py, one per line of jobfile:
  %   pf  <case file> <lambda> <output file>   runpf with pf.enforce_q_lims = 1
  %   cpf <case file> <qlims>  <output file>   runcpf to the nose, cpf.enforce_q_lims = qlims
  %   dcopf <case file> 0      <output file>   rundcopf at the case's own data
  % In both the reference generator is unlimited (Q limits +-1e9), as in gridnet's single-slack
  % models; pf scales the loads and the non-reference generation by lambda, cpf runs from the case to
  % twice them. Output: a first row [success, lambda, 0, 0, 0, 0] (lambda: the given one, or 1 plus
  % runcpf's max_lam), then per generator [bus, QG, QMIN, QMAX, VM at its bus, VG], then per bus
  % [VM, VA, 0, 0, 0, 0]. For dcopf: a first row [success, f, 0, 0, 0, 0], per generator [bus, PG, 0, 0,
  % 0, 0], per bus [LAM_P, VA, 0, 0, 0, 0].
  for d = {'lib', 'mips/lib', 'mp-opt-model/lib', 'mptest/lib'}
    addpath(fullfile(mproot, d{1}));
  end
  define_constants;
  fid = fopen(jobfile); jobs = {};
  while true, l = fgetl(fid); if ~ischar(l), break; end; if ~isempty(strtrim(l)), jobs{end+1} = l; end; end
  fclose(fid);
  for k = 1:numel(jobs)
    p = strsplit(jobs{k}, char(9));
    kind = p{1}; casefile = p{2}; value = str2double(p{3}); out = p{4};
    mpc = loadcase(casefile);
    ref = find(mpc.bus(:, BUS_TYPE) == REF);
    atref = mpc.gen(:, GEN_BUS) == mpc.bus(ref, BUS_I);
    mpc.gen(atref, QMAX) = 1e9; mpc.gen(atref, QMIN) = -1e9;
    if strcmp(kind, 'pf')
      mpc.bus(:, [PD QD]) = mpc.bus(:, [PD QD]) * value;
      mpc.gen(~atref, PG) = mpc.gen(~atref, PG) * value;
      opt = mpoption('verbose', 0, 'out.all', 0, 'pf.enforce_q_lims', 1, 'pf.tol', 1e-10, 'pf.nr.max_it', 30);
      r = runpf(mpc, opt); lam = value;
    elseif strcmp(kind, 'dcopf')
      r = rundcopf(loadcase(casefile), mpoption('verbose', 0, 'out.all', 0));
      g = r.gen;
      M = [r.success, r.f, 0, 0, 0, 0; g(:, [GEN_BUS PG]), zeros(size(g, 1), 4); r.bus(:, [LAM_P VA]), zeros(size(r.bus, 1), 4)];
      dlmwrite(out, M, 'precision', '%.17g');
      continue
    else
      target = mpc;
      target.bus(:, [PD QD]) = 2 * mpc.bus(:, [PD QD]);
      target.gen(~atref, PG) = 2 * mpc.gen(~atref, PG);
      opt = mpoption('verbose', 0, 'out.all', 0, 'cpf.stop_at', 'NOSE', 'cpf.enforce_q_lims', value, 'cpf.step', 0.02, ...
                     'cpf.adapt_step', 1, 'cpf.step_min', 1e-6, 'cpf.adapt_step_tol', 1e-6, 'cpf.nose_tol', 1e-8);
      r = runcpf(mpc, target, opt); lam = 1 + r.cpf.max_lam;
    end
    g = r.gen; b = arrayfun(@(x) find(r.bus(:, BUS_I) == x), g(:, GEN_BUS));
    M = [r.success, lam, 0, 0, 0, 0; g(:, [GEN_BUS QG QMIN QMAX]), r.bus(b, VM), g(:, VG); r.bus(:, [VM VA]), zeros(size(r.bus, 1), 4)];
    dlmwrite(out, M, 'precision', '%.17g');
  end
end
