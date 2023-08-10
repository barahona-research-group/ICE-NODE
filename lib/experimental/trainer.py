# class Trainer2LR(Trainer):

#     def init_from_loaded_optstate(self, optstate, model, iters):
#         optstate = (jopt.pack_optimizer_state(optstate[0]),
#                     jopt.pack_optimizer_state(optstate[1]))
#         opt, _ = self.init_opt(model, iters=iters)

#         return opt, optstate

#     def serializable_optstate(self, optstate):
#         _, optstate = optstate
#         return (jopt.unpack_optimizer_state(optstate[0]),
#                 jopt.unpack_optimizer_state(optstate[1]))

#     def init_opt(self, model: AbstractModel, iters: int):
#         decay_rate = self.decay_rate
#         if not (isinstance(decay_rate, list) or isinstance(decay_rate, tuple)):
#             decay_rate = (decay_rate, decay_rate)

#         opt1_i, opt1_u, opt1_p = opts[self.opt](self.lr_schedule(self.lr[0],
#                                                                  decay_rate[0],
#                                                                  iters=iters))
#         opt2_i, opt2_u, opt2_p = opts[self.opt](self.lr_schedule(self.lr[1],
#                                                                  decay_rate[1],
#                                                                  iters=iters))
#         m1, m2 = model.emb_dyn_partition(model)
#         m1 = eqx.filter(m1, eqx.is_inexact_array)
#         m2 = eqx.filter(m2, eqx.is_inexact_array)
#         opt1_s = opt1_i(m1)
#         opt2_s = opt2_i(m2)
#         opt1 = opt1_u, opt1_p
#         opt2 = opt2_u, opt2_p
#         return (opt1, opt2), (opt1_s, opt2_s)

#     def step_optimizer(self, step: int, opt_state: Any, model: AbstractModel,
#                        patients: Patients):
#         (opt1, opt2), (opt1_s, opt2_s) = opt_state
#         opt1_u, opt1_p = opt1
#         opt2_u, opt2_p = opt2

#         grad_f = eqx.filter_grad(self.loss, has_aux=True)
#         grads, aux = grad_f(model, patients)
#         g1, g2 = model.emb_dyn_partition(grads)

#         opt1_s = opt1_u(step, g1, opt1_s)
#         opt2_s = opt2_u(step, g2, opt2_s)

#         new_params = model.emb_dyn_merge(opt1_p(opt1_s), opt2_p(opt2_s))
#         new_model = self.update_model(model, new_params)

#         return ((opt1, opt2), (opt1_s, opt2_s)), new_model, aux




# class ODETrainer2LR(ODETrainer, Trainer2LR):
#     pass


# class InTrainer2LR(InTrainer, Trainer2LR):
#     pass
