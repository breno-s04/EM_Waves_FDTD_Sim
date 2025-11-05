import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os
import shutil
from pathlib import Path
import subprocess
import sys

class FDTD2D:
    """
    Simulador de Ondas Eletromagnéticas 2D usando o Método FDTD (Yee Algorithm)
    """
    
    def __init__(self, nx, ny, dx, dy, courant_factor=0.5):
        """
        Inicializa o simulador FDTD 2D
        
        Parâmetros:
        -----------
        nx, ny : int
            Dimensões da grade de simulação
        dx, dy : float
            Espaçamento espacial da grade (em metros)
        courant_factor : float
            Fator de segurança para a condição CFL (< 1.0)
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Constantes físicas
        self.c0 = 3e8  # Velocidade da luz no vácuo (m/s)
        self.mu0 = 4 * np.pi * 1e-7  # Permeabilidade magnética do vácuo
        self.eps0 = 8.854187817e-12  # Permissividade elétrica do vácuo
        
        # Cálculo do passo de tempo pela condição CFL
        self.dt = courant_factor / (self.c0 * np.sqrt(1/dx**2 + 1/dy**2))
        
        # Inicialização dos campos (componente TMz: Ez, Hx, Hy)
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))
        
        # Propriedades dos materiais (permissividade relativa)
        self.eps_r = np.ones((nx, ny))  # Inicialmente tudo é vácuo
        
        # Camadas PML (Perfectly Matched Layer)
        self.pml_thickness = 20
        self.setup_pml()
        
        # Controle de tempo
        self.time_step = 0
        self.current_time = 0.0
        
        # Histórico de energia para análise
        self.energy_history = []
        self.time_history = []
        
    def setup_pml(self):
        """Configura as camadas PML nas bordas da grade com gradiente cúbico"""
        # Parâmetros do PML otimizados
        m = 3.0  # Ordem do gradiente
        sigma_max = 0.7 * (m + 1) / (150 * np.pi * self.dx)
        kappa_max = 15.0
        alpha_max = 0.05
        
        # Arrays de coeficientes PML
        self.sigma_e = np.zeros((self.nx, self.ny))
        self.sigma_h = np.zeros((self.nx, self.ny))
        self.kappa_e = np.ones((self.nx, self.ny))
        self.kappa_h = np.ones((self.nx, self.ny))
        
        # Camadas PML (Perfectly Matched Layer)
        self.pml_thickness = 10
        self.setup_pml()
        
        # Controle de tempo
        self.time_step = 0
        self.current_time = 0.0
        
    def setup_pml(self):
        """Configura as camadas PML nas bordas da grade com gradiente cúbico"""
        # Parâmetros do PML otimizados
        m = 3.0  # Ordem do gradiente
        sigma_max = 0.7 * (m + 1) / (150 * np.pi * self.dx)
        kappa_max = 15.0
        alpha_max = 0.05
        
        # Arrays de coeficientes PML
        self.sigma_e = np.zeros((self.nx, self.ny))
        self.sigma_h = np.zeros((self.nx, self.ny))
        self.kappa_e = np.ones((self.nx, self.ny))
        self.kappa_h = np.ones((self.nx, self.ny))
        
        # Criar perfis PML para as bordas
        for i in range(self.pml_thickness):
            # Distância normalizada da borda (0 na borda, 1 no interior)
            depth = (self.pml_thickness - i) / self.pml_thickness
            
            # Perfis graduais (polinomial)
            sigma_val = sigma_max * (depth ** m)
            kappa_val = 1 + (kappa_max - 1) * (depth ** m)
            
            # Aplicar nas quatro bordas
            # Borda esquerda (x = 0)
            self.sigma_e[i, :] = np.maximum(self.sigma_e[i, :], sigma_val)
            self.kappa_e[i, :] = np.maximum(self.kappa_e[i, :], kappa_val)
            
            # Borda direita (x = nx-1)
            self.sigma_e[-(i+1), :] = np.maximum(self.sigma_e[-(i+1), :], sigma_val)
            self.kappa_e[-(i+1), :] = np.maximum(self.kappa_e[-(i+1), :], kappa_val)
            
            # Borda inferior (y = 0)
            self.sigma_e[:, i] = np.maximum(self.sigma_e[:, i], sigma_val)
            self.kappa_e[:, i] = np.maximum(self.kappa_e[:, i], kappa_val)
            
            # Borda superior (y = ny-1)
            self.sigma_e[:, -(i+1)] = np.maximum(self.sigma_e[:, -(i+1)], sigma_val)
            self.kappa_e[:, -(i+1)] = np.maximum(self.kappa_e[:, -(i+1)], kappa_val)
        
        self.sigma_h = self.sigma_e * self.mu0 / self.eps0
        self.kappa_h = self.kappa_e
        
        # Calcular coeficientes de atualização do PML
        self.update_pml_coefficients()
    
    def update_pml_coefficients(self):
        """Calcula os coeficientes de atualização considerando PML e materiais"""
        # Coeficientes para o campo E
        den_e = self.kappa_e * self.eps0 * self.eps_r + 0.5 * self.sigma_e * self.dt
        self.Ca = (self.kappa_e * self.eps0 * self.eps_r - 0.5 * self.sigma_e * self.dt) / den_e
        self.Cb = (self.dt / den_e) / self.dx
        
        # Coeficientes para o campo H
        den_h = self.kappa_h * self.mu0 + 0.5 * self.sigma_h * self.dt
        self.Da = (self.kappa_h * self.mu0 - 0.5 * self.sigma_h * self.dt) / den_h
        self.Db = (self.dt / den_h) / self.dy
        self.Dc = (self.dt / den_h) / self.dx
    
    def add_dielectric_box(self, x1, y1, x2, y2, eps_r):
        """
        Adiciona uma região retangular com propriedades dielétricas
        
        Parâmetros:
        -----------
        x1, y1, x2, y2 : int
            Coordenadas do retângulo na grade
        eps_r : float
            Permissividade relativa do material
        """
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        x1 = max(0, min(x1, self.nx-1))
        x2 = max(0, min(x2, self.nx-1))
        y1 = max(0, min(y1, self.ny-1))
        y2 = max(0, min(y2, self.ny-1))
        
        self.eps_r[x1:x2+1, y1:y2+1] = eps_r
        self.update_pml_coefficients()
    
    def add_dielectric_circle(self, cx, cy, radius, eps_r):
        """
        Adiciona uma região circular com propriedades dielétricas
        
        Parâmetros:
        -----------
        cx, cy : int
            Centro do círculo na grade
        radius : int
            Raio do círculo em pixels
        eps_r : float
            Permissividade relativa do material
        """
        y, x = np.ogrid[:self.nx, :self.ny]
        mask = (x - cy)**2 + (y - cx)**2 <= radius**2
        self.eps_r[mask] = eps_r
        self.update_pml_coefficients()
    
    def add_gaussian_source(self, x, y, t0, spread):
        """
        Adiciona uma fonte gaussiana em um ponto específico
        
        Parâmetros:
        -----------
        x, y : int
            Posição da fonte na grade
        t0 : float
            Tempo central do pulso
        spread : float
            Largura do pulso gaussiano
        """
        pulse = np.exp(-((self.current_time - t0) / spread) ** 2)
        self.Ez[x, y] += pulse
    
    def add_sinusoidal_source(self, x, y, frequency):
        """
        Adiciona uma fonte senoidal contínua
        
        Parâmetros:
        -----------
        x, y : int
            Posição da fonte na grade
        frequency : float
            Frequência da fonte (Hz)
        """
        omega = 2 * np.pi * frequency
        self.Ez[x, y] += np.sin(omega * self.current_time)
    
    def calculate_energy(self):
        """
        Calcula a energia eletromagnética total no domínio
        
        Retorna:
        --------
        float : Energia total (J)
        """
        # Energia elétrica: (1/2) * ε₀ * εᵣ * E²
        electric_energy = 0.5 * self.eps0 * np.sum(self.eps_r * self.Ez**2) * self.dx * self.dy
        
        # Energia magnética: (1/2) * μ₀ * H²
        magnetic_energy = 0.5 * self.mu0 * np.sum(self.Hx**2 + self.Hy**2) * self.dx * self.dy
        
        return electric_energy + magnetic_energy
    
    def update_H(self):
        """Atualiza os campos magnéticos Hx e Hy"""
        # Atualiza Hx
        self.Hx[:, :-1] = self.Da[:, :-1] * self.Hx[:, :-1] - \
                          self.Db[:, :-1] * (self.Ez[:, 1:] - self.Ez[:, :-1])
        
        # Atualiza Hy
        self.Hy[:-1, :] = self.Da[:-1, :] * self.Hy[:-1, :] + \
                          self.Db[:-1, :] * (self.Ez[1:, :] - self.Ez[:-1, :])
    
    def update_E(self):
        """Atualiza o campo elétrico Ez"""
        self.Ez[1:, 1:] = self.Ca[1:, 1:] * self.Ez[1:, 1:] + \
                          self.Cb[1:, 1:] * ((self.Hy[1:, 1:] - self.Hy[:-1, 1:]) - \
                                              (self.Hx[1:, 1:] - self.Hx[1:, :-1]))
    
    def step(self, source_func=None):
        """
        Executa um passo de tempo da simulação
        
        Parâmetros:
        -----------
        source_func : callable, opcional
            Função que adiciona fontes: source_func(simulator)
        """
        self.update_H()
        self.update_E()
        
        if source_func is not None:
            source_func(self)
        
        # Calcular e armazenar energia
        energy = self.calculate_energy()
        self.energy_history.append(energy)
        self.time_history.append(self.current_time)
        
        self.time_step += 1
        self.current_time += self.dt


class SimulationUI:
    """Interface de usuário para configuração da simulação"""
    
    @staticmethod
    def get_int_input(prompt, default=None, min_val=None, max_val=None):
        """Obtém entrada inteira do usuário com validação"""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} (padrão: {default}): ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                value = int(user_input)
                
                if min_val is not None and value < min_val:
                    print(f"Erro: Valor deve ser >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Erro: Valor deve ser <= {max_val}")
                    continue
                
                return value
            except ValueError:
                print("Erro: Por favor, digite um número inteiro válido.")
    
    @staticmethod
    def get_float_input(prompt, default=None, min_val=None, max_val=None):
        """Obtém entrada de ponto flutuante do usuário com validação"""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} (padrão: {default}): ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                value = float(user_input)
                
                if min_val is not None and value < min_val:
                    print(f"Erro: Valor deve ser >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Erro: Valor deve ser <= {max_val}")
                    continue
                
                return value
            except ValueError:
                print("Erro: Por favor, digite um número válido.")
    
    @staticmethod
    def get_yes_no(prompt, default='n'):
        """Obtém resposta sim/não do usuário"""
        while True:
            response = input(f"{prompt} (s/n, padrão: {default}): ").strip().lower()
            if not response:
                response = default
            if response in ['s', 'sim', 'y', 'yes']:
                return True
            elif response in ['n', 'nao', 'não', 'no']:
                return False
            else:
                print("Por favor, responda com 's' (sim) ou 'n' (não).")


def run_simulation():
    """Função principal que executa a simulação interativa"""
    
    print("=" * 70)
    print("SIMULADOR DE ONDAS ELETROMAGNÉTICAS - MÉTODO FDTD 2D")
    print("Projeto de Computação Científica - UNICAMP")
    print("Autor: Breno Santimaria Sertori")
    print("=" * 70)
    print()
    
    ui = SimulationUI()
    
    # Configuração da grade
    print("CONFIGURAÇÃO DA GRADE DE SIMULAÇÃO")
    print("-" * 70)
    nx = ui.get_int_input("Tamanho da grade em X (pixels)", default=200, min_val=50, max_val=1000)
    ny = ui.get_int_input("Tamanho da grade em Y (pixels)", default=200, min_val=50, max_val=1000)
    
    # Escala física
    print("\nESCALA FÍSICA")
    print("-" * 70)
    wavelength = ui.get_float_input("Comprimento de onda desejado (nm)", default=500, min_val=100)
    wavelength_m = wavelength * 1e-9  # Converte para metros
    
    # Resolução: pelo menos 20 pixels por comprimento de onda
    dx = wavelength_m / 20
    dy = dx
    
    print(f"\nResolução espacial calculada: dx = dy = {dx*1e9:.2f} nm")
    print(f"Dimensão física da grade: {nx*dx*1e9:.1f} nm x {ny*dy*1e9:.1f} nm")
    
    # Criar simulador
    sim = FDTD2D(nx, ny, dx, dy, courant_factor=0.5)
    
    print(f"Passo de tempo (dt): {sim.dt*1e15:.3f} fs")
    print(f"Condição CFL satisfeita: c*dt*sqrt(1/dx² + 1/dy²) = {sim.c0*sim.dt*np.sqrt(1/dx**2 + 1/dy**2):.3f}")
    
    # Configuração da fonte
    print("\nCONFIGURAÇÃO DA FONTE")
    print("-" * 70)
    print("1 - Fonte pontual (pulso gaussiano)")
    print("2 - Fonte senoidal contínua")
    source_type = ui.get_int_input("Escolha o tipo de fonte", default=2, min_val=1, max_val=2)
    
    source_x = ui.get_int_input(f"Posição X da fonte (0-{nx-1})", default=nx//2, min_val=0, max_val=nx-1)
    source_y = ui.get_int_input(f"Posição Y da fonte (0-{ny-1})", default=ny//2, min_val=0, max_val=ny-1)
    
    if source_type == 1:
        frequency = ui.get_float_input("Frequência central (THz)", default=3e5//wavelength, min_val=1,max_val=3e5//wavelength)
        frequency_hz = frequency * 1e12
        spread = 1 / frequency_hz
        t0 = 3 * spread
        
        def source_func(s):
            if s.current_time <= t0 + 3*spread:
                s.add_gaussian_source(source_x, source_y, t0, spread)
    else:
        frequency = ui.get_float_input("Frequência (THz)", default=3e5//wavelength, min_val=1,max_val=3e5//wavelength)
        frequency_hz = frequency * 1e12
        
        def source_func(s):
            s.add_sinusoidal_source(source_x, source_y, frequency_hz)
    
    # Configuração de materiais
    print("\nCONFIGURAÇÃO DE MATERIAIS DIELÉTRICOS")
    print("-" * 70)
    add_materials = ui.get_yes_no("Deseja adicionar materiais dielétricos?", default='n')
    
    materials_list = []
    if add_materials:
        num_materials = ui.get_int_input("Quantos objetos dielétricos?", default=1, min_val=1, max_val=10)
        
        for i in range(num_materials):
            print(f"\nMaterial {i+1}:")
            print("1 - Retângulo")
            print("2 - Círculo")
            shape = ui.get_int_input("Forma do objeto", default=1, min_val=1, max_val=2)
            
            eps_r = ui.get_float_input("Permissividade relativa (εr)", default=2.25, min_val=1.0)
            
            if shape == 1:
                x1 = ui.get_int_input(f"X inicial (0-{nx-1})", min_val=0, max_val=nx-1)
                y1 = ui.get_int_input(f"Y inicial (0-{ny-1})", min_val=0, max_val=ny-1)
                x2 = ui.get_int_input(f"X final (0-{nx-1})", min_val=0, max_val=nx-1)
                y2 = ui.get_int_input(f"Y final (0-{ny-1})", min_val=0, max_val=ny-1)
                sim.add_dielectric_box(x1, y1, x2, y2, eps_r)
                materials_list.append(('box', x1, y1, x2, y2, eps_r))
            else:
                cx = ui.get_int_input(f"Centro X (0-{nx-1})", min_val=0, max_val=nx-1)
                cy = ui.get_int_input(f"Centro Y (0-{ny-1})", min_val=0, max_val=ny-1)
                radius = ui.get_int_input("Raio (pixels)", min_val=1, max_val=min(nx, ny)//4)
                sim.add_dielectric_circle(cx, cy, radius, eps_r)
                materials_list.append(('circle', cx, cy, radius, eps_r))
    
    # Configuração da simulação
    print("\nCONFIGURAÇÃO DA EXECUÇÃO")
    print("-" * 70)
    num_steps = ui.get_int_input("Número de passos de tempo", default=300, min_val=50, max_val=5000)
    
    save_video = ui.get_yes_no("Deseja salvar o vídeo da simulação?", default='s')
    video_filename = "simulacao_fdtd.mp4"
    if save_video:
        video_filename = input(f"Nome do arquivo de vídeo (padrão: {video_filename}): ").strip()
        if not video_filename:
            video_filename = "simulacao_fdtd.mp4"
        if not video_filename.endswith('.mp4'):
            video_filename += '.mp4'
    
    # Executar simulação com visualização
    print("\n" + "=" * 70)
    print("INICIANDO SIMULAÇÃO")
    print("=" * 70)
    
    # Diretório temporário para frames
    temp_dir = None
    if save_video:
        temp_dir = Path('temp_frames')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        print("Modo: Simulação sem visualização (mais rápido)")
        print(f"Salvando frames em: {temp_dir}")
        
        # Perguntar sobre taxa de frames
        save_every = 1
        if num_steps > 500:
            if ui.get_yes_no("Simulação longa detectada. Salvar apenas 1 frame a cada 2 passos? (economiza espaço)", default='s'):
                save_every = 2
        
        print(f"Salvando 1 frame a cada {save_every} passo(s)")
        print()
        
        # Criar figura apenas para salvar frames (sem mostrar)
        plt.ioff()  # Desliga modo interativo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Criar colorbars fixas (apenas uma vez)
        # Placeholder inicial
        im1 = ax1.imshow(sim.Ez.T, cmap='RdBu', origin='lower', 
                         vmin=-0.1, vmax=0.1, interpolation='bilinear')
        ax1.set_title('Campo Elétrico (Ez)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Ez (V/m)')
        
        # Adicionar retângulo mostrando região de simulação (dentro do PML)
        pml = sim.pml_thickness
        pml_rect = Rectangle((pml, pml), nx-2*pml, ny-2*pml, 
                             fill=False, edgecolor='yellow', linewidth=1.5, 
                             linestyle='--', label='Região de simulação')
        ax1.add_patch(pml_rect)
        ax1.legend(loc='upper right', fontsize=8)
        
        im2 = ax2.imshow(sim.eps_r.T, cmap='viridis', origin='lower', 
                         vmin=1.0, vmax=np.max(sim.eps_r))
        ax2.set_title('Permissividade Relativa (εr)')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        cbar2 = plt.colorbar(im2, ax=ax2, label='εr')
        
        # Texto de status
        time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        frame_count = 0
        
        # Determinar escala de cores fixa baseada em uma estimativa
        # Rodar alguns passos iniciais para estimar o máximo
        print("Estimando amplitude máxima do campo...")
        max_field_estimate = 0
        for step in range(min(50, num_steps)):
            sim.step(source_func)
            max_field_estimate = max(max_field_estimate, np.max(np.abs(sim.Ez)))
        
        # Usar a estimativa como máximo fixo
        fixed_vmax = max_field_estimate
        if fixed_vmax < 1e-10:
            fixed_vmax = 1.0  # Valor padrão se ainda não houver campo
        
        print(f"Escala de cores fixa: ±{fixed_vmax:.2e} V/m")
        print()
        
        # Resetar simulação para começar do zero
        sim.time_step = 0
        sim.current_time = 0.0
        sim.Ez = np.zeros((nx, ny))
        sim.Hx = np.zeros((nx, ny))
        sim.Hy = np.zeros((nx, ny))
        sim.energy_history = []
        sim.time_history = []
        
        # Atualizar escala do imshow com valores fixos
        im1.set_clim(vmin=-fixed_vmax, vmax=fixed_vmax)
        
        # Loop de simulação sem animação
        for step in range(num_steps):
            sim.step(source_func)
            
            # Salvar frame a cada N passos
            if step % save_every == 0:
                # Atualizar apenas os dados (escala fixa, não muda)
                im1.set_data(sim.Ez.T)
                
                # Atualizar texto
                time_text.set_text(f'Passo: {sim.time_step}\nTempo: {sim.current_time*1e15:.2f} fs')
                
                # Salvar frame
                fig.savefig(temp_dir / f'frame_{frame_count:04d}.png', dpi=100, 
                           bbox_inches='tight', pad_inches=0.1)
                frame_count += 1
            
            # Progresso no terminal
            if step % 50 == 0:
                print(f"Progresso: {step}/{num_steps} passos ({100*step/num_steps:.1f}%)")
        
        plt.close(fig)
        print(f"\n✓ Simulação concluída! {num_steps} passos processados, {frame_count} frames salvos.")
        
    else:
        # Modo com visualização ao vivo (sem salvar vídeo)
        print("Modo: Visualização ao vivo")
        print()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Configurar plot do campo Ez
        im1 = ax1.imshow(sim.Ez.T, cmap='RdBu', origin='lower', 
                         vmin=-0.1, vmax=0.1, interpolation='bilinear')
        ax1.set_title('Campo Elétrico (Ez)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Ez (V/m)')
        
        # Adicionar retângulo mostrando região de simulação (dentro do PML)
        pml = sim.pml_thickness
        pml_rect = Rectangle((pml, pml), nx-2*pml, ny-2*pml, 
                             fill=False, edgecolor='yellow', linewidth=1.5, 
                             linestyle='--', label='Região de simulação')
        ax1.add_patch(pml_rect)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Configurar plot da permissividade
        im2 = ax2.imshow(sim.eps_r.T, cmap='viridis', origin='lower', 
                         vmin=1.0, vmax=np.max(sim.eps_r))
        ax2.set_title('Permissividade Relativa (εr)')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='εr')
        
        time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', 
                            facecolor='wheat', alpha=0.8))
        
        def animate(frame):
            sim.step(source_func)
            
            # Atualizar visualização
            im1.set_data(sim.Ez.T)
            im1.set_clim(vmin=-np.max(np.abs(sim.Ez))*0.8, 
                         vmax=np.max(np.abs(sim.Ez))*0.8)
            
            time_text.set_text(f'Passo: {sim.time_step}\nTempo: {sim.current_time*1e15:.2f} fs')
            
            if frame % 50 == 0:
                print(f"Progresso: {frame}/{num_steps} passos ({100*frame/num_steps:.1f}%)")
            
            return [im1, time_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=num_steps, 
                                      interval=20, blit=True, repeat=False)
        
        plt.tight_layout()
        plt.show()
    
    # Análise pós-simulação: Gráfico de Energia
    print("\n" + "=" * 70)
    print("ANÁLISE DOS RESULTADOS")
    print("=" * 70)
    
    if len(sim.energy_history) > 10:
        energies = np.array(sim.energy_history)
        times = np.array(sim.time_history)
        
        max_energy = np.max(energies)
        final_energy = energies[-1]
        energy_decay = (max_energy - final_energy) / max_energy * 100 if max_energy > 0 else 0
        
        print(f"Energia máxima: {max_energy:.3e} J")
        print(f"Energia final: {final_energy:.3e} J")
        print(f"Decaimento de energia (absorção PML): {energy_decay:.2f}%")
        
        # Perguntar se deseja ver gráfico de energia
        if ui.get_yes_no("\nDeseja ver gráfico de energia vs tempo?", default='s'):
            fig2, (ax_e1, ax_e2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Energia vs tempo (escala linear)
            ax_e1.plot(times * 1e15, energies, 'b-', linewidth=2)
            ax_e1.set_xlabel('Tempo (fs)')
            ax_e1.set_ylabel('Energia (J)')
            ax_e1.set_title('Energia Eletromagnética Total vs Tempo')
            ax_e1.grid(True, alpha=0.3)
            
            # Energia normalizada (escala log)
            if max_energy > 0:
                normalized_energy = energies / max_energy
                ax_e2.semilogy(times * 1e15, normalized_energy, 'r-', linewidth=2)
                ax_e2.set_xlabel('Tempo (fs)')
                ax_e2.set_ylabel('Energia Normalizada')
                ax_e2.set_title('Energia Normalizada (Escala Logarítmica)')
                ax_e2.grid(True, alpha=0.3, which='both')
                ax_e2.set_ylim(bottom=1e-3)
            
            plt.tight_layout()
            plt.show()
    
    # Compilar vídeo com FFmpeg
    if save_video and temp_dir is not None:
        print("\n" + "=" * 70)
        print("COMPILANDO VÍDEO COM FFMPEG...")
        print("=" * 70)
        
        try:
            # Verificar se FFmpeg está instalado
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            # Compilar frames em vídeo (com filtro para garantir dimensões pares)
            cmd = [
                'ffmpeg', '-y',
                '-framerate', '30',
                '-i', str(temp_dir / 'frame_%04d.png'),
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Força dimensões pares
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                video_filename
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"\n✓ Vídeo salvo com sucesso: {video_filename}")
                file_size = Path(video_filename).stat().st_size / (1024 * 1024)
                print(f"  Tamanho do arquivo: {file_size:.2f} MB")
            else:
                print(f"\n✗ Erro ao compilar vídeo:")
                print(result.stderr)
        
        except FileNotFoundError:
            print("\n✗ FFmpeg não encontrado!")
            print("  Por favor, instale o FFmpeg para salvar vídeos.")
            print("  Os frames foram salvos em:", temp_dir)
        except Exception as e:
            print(f"\n✗ Erro ao processar vídeo: {e}")
        
        finally:
            # Limpar arquivos temporários
            if ui.get_yes_no("\nRemover frames temporários?", default='s'):
                shutil.rmtree(temp_dir)
                print("Frames temporários removidos.")
    
    print("\n" + "=" * 70)
    print("SIMULAÇÃO CONCLUÍDA")
    print("=" * 70)
    print(f"Total de passos simulados: {sim.time_step}")
    print(f"Tempo físico total: {sim.current_time*1e15:.2f} femtosegundos")
    print("\nObrigado por usar o simulador FDTD!")


if __name__ == "__main__":
    run_simulation()