// Regenerate the JS reference values used by tests/test_kinematics.py and tests/test_calibration.py.
// Evaluates the real code blocks from the rebot-armd Web cockpit (IK.fk, IK.solve, solveRigid).
// Usage: node tools/gen_web_fixtures.mjs [path/to/rebot-armd/src/api/index.html]
import fs from 'node:fs';

const path = process.argv[2] || '../rebot-armd/src/api/index.html';
const html = fs.readFileSync(path, 'utf8');
const cut = (a, b) => {
  const i = html.indexOf(a), j = html.indexOf(b, i);
  if (i < 0 || j < 0) throw new Error(`block not found: ${a}`);
  return html.slice(i, j);
};
const ikSrc = cut('/* IK-BEGIN */', '/* IK-END */');
const rigidSrc = cut('function maxEigVec4(A)', '// freshest marker');
// eslint-disable-next-line no-new-func
const lib = new Function(`${ikSrc}\n${rigidSrc}\nreturn { IK, solveRigid };`)();

const fkCases = [
  [0, 0, 0, 0], [0.3, -1.0, -1.5, 0.5], [-0.8, -0.4, -2.2, 1.1], [1.7, -2.0, -0.6, -0.9],
  [2.9, -0.1, -2.9, 1.5], [-2.5, -1.4, -1.9, 0.0],
];
const fk = fkCases.map((q) => ({ q, out: lib.IK.fk(q) }));

const ikCases = [
  { q: [0.2, -1.0, -2.2, 0.8], prev: [0.25, -0.9, -2.1, 0.7] },
  { q: [-0.6, -0.5, -1.8, 1.2], prev: [-0.5, -0.6, -1.7, 1.0] },
  { q: [1.2, -1.6, -1.0, -0.3], prev: [1.0, -1.5, -1.1, -0.2] },
];
const ik = ikCases.map(({ q, prev }) => {
  const p = lib.IK.fk(q);
  return { target: p, prev, out: lib.IK.solve(p, { prev }) };
});

// deterministic noisy point set: p_base = R p_cam + t + noise
const R0 = (a) => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
let seed = 7;
const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
const Rt = R0(0.7), tt = [0.41, -0.27, 0.63];
const cam = [], base = [];
for (let i = 0; i < 9; i++) {
  const pc = [rnd() * 0.4, rnd() * 0.3, 0.5 + rnd() * 0.3];
  const pb = [0, 1, 2].map((r) => Rt[r][0] * pc[0] + Rt[r][1] * pc[1] + Rt[r][2] * pc[2] + tt[r] + rnd() * 0.004);
  cam.push(pc); base.push(pb);
}
const rigid = { cam, base, out: lib.solveRigid(cam, base) };
const degenerate = lib.solveRigid([[0, 0, 0.5], [0.1, 0, 0.5], [0.2, 0, 0.5]], [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]);

console.log(JSON.stringify({ fk, ik, rigid, degenerate }, null, 1));
