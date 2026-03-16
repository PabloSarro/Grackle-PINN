import torch

def physics_loss(model, x):
    # x contains [T_init, HI, HII, dt]
    x.requires_grad = True
    T_pred = model(x)
    
    # Use Autograd to get dT/dt (derivative of prediction w.r.t time)
    # This is the "Informed" part of the PINN
    grads = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    dT_dt_pred = grads[:, 3] # gradient with respect to dt (index 3)

    # Simplified physical cooling rate (placeholder for Grackle logic)
    # Residual = pred_rate - physical_rate
    physics_residual = dT_dt_pred - (-1.0e-20 * x[:, 0]) # Simple linear cooling example
    return torch.mean(physics_residual**2)