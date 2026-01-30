## This is the base format for kube + kustomize + github-ci

### ArgoCD local test
1. create kubectl argocd
```bash
kubectl create namespace myargocd
```

2. ArgoCD install
```bash
kubectl apply -n myargocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

3. check argocd admin setting
```bash
kubectl get secret argocd-initial-admin-secret -n myargocd -o yaml
```

4. Option: NodePort
```bash
kubectl patch svc argocd-server -n myargocd -p '{"spec": {"type": "NodePort"}}'
```
```bash
kubectl get svc argocd-server -n myargocd
```