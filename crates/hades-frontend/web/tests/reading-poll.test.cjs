/* Control response ordering around the production polling callback. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../reading.js'),'utf8');
const callback=source.slice(source.indexOf('async function pollReading(){'),source.indexOf('setInterval(pollReading,3000);'));
function fixture(){
 let respond;
 const ctx={state:{id:'one',revision:4},busy:false,document:{hidden:false},renders:0,
   api:()=>new Promise(resolve=>{respond=resolve}),renderState:()=>ctx.renders++,status:()=>{}};
 const poll=vm.runInNewContext(callback+';pollReading',ctx);
 return {ctx,poll,reply:(revision=5)=>respond({state:{id:'one',revision},connectivity:[]})};
}
test('late poll cannot undo a mutation or replace a different reading',async()=>{
 for(const change of [c=>{c.state.revision=6},c=>{c.busy=true},c=>{c.state={id:'two',revision:0}}]){
  const f=fixture(),pending=f.poll();change(f.ctx);const expected={...f.ctx.state};f.reply();await pending;
  assert.deepEqual(f.ctx.state,expected);assert.equal(f.ctx.renders,0);
 }
});
test('only newer idle poll results render',async()=>{
 for(const revision of [3,4,5]){
  const f=fixture(),pending=f.poll();f.reply(revision);await pending;
  assert.equal(f.ctx.state.revision,Math.max(4,revision));assert.equal(f.ctx.renders,revision>4?1:0);
 }
});
