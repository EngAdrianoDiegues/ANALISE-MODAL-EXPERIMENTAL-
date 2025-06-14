# Bibliotecas

from kivy.app import App
from kivy.uix.filechooser import FileChooserIconView
from scipy.optimize import curve_fit
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import pyuff
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from sdypy import EMA as pyEMA
import os
from kivy.uix.filechooser import FileChooserListView
import re
from kivy.uix.scrollview import ScrollView
from matplotlib.animation import FuncAnimation
from kivy.uix.textinput import TextInput

# Cria a classe do aplicativo

class AME(App):

# Função para criar os botões

    def build(self):
        # Layout principal que permite sobreposição de imagem e botões
        self.layout = FloatLayout()

        # Imagem de fundo
        imagem_fundo = Image(
            source='foto_GVA.png',
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 1),
            pos_hint={'x': 0, 'y': 0}
        )
        self.layout.add_widget(imagem_fundo)

        # Layout horizontal para os botões na parte inferior da tela
        layout_botoes = BoxLayout(
            orientation='horizontal',
            spacing=20,
            size_hint=(None, None),
            size=(660, 50),
            pos_hint={'center_x': 0.5, 'y': 0.1}
        )

        # Botão para inserir o arquivo FRF
        botao_inserir_frf = Button(
            text="Inserir FRF",
            size_hint=(None, None),
            size=(200, 50)
        )
        botao_inserir_frf.bind(on_release=self.procurar)

        # Botão para calcular os parâmetros (inicialmente desativado)
        botao_calcular_parametros = Button(
            text="Calcular os Parâmetros",
            size_hint=(None, None),
            size=(200, 50),
            disabled=True
        )
        botao_calcular_parametros.bind(on_release=self.mostrar_novo_layout)
        self.botao_calcular_parametros = botao_calcular_parametros

        # Botão para exibir o tutorial
        botao_tutorial = Button(
            text="Tutorial",
            size_hint=(None, None),
            size=(200, 50)
        )
        botao_tutorial.bind(on_release=self.exibir_mensagem)

        # Adiciona os botões ao layout de botões
        layout_botoes.add_widget(botao_inserir_frf)
        layout_botoes.add_widget(botao_calcular_parametros)
        layout_botoes.add_widget(botao_tutorial)

        # Adiciona o layout de botões ao layout principal
        self.layout.add_widget(layout_botoes)

        return self.layout

    # Função para prorcurar a pasta
    def procurar(self, instancia):
        # Cria o seletor de arquivos configurado para seleção de pastas
        self.seletor_pasta = FileChooserListView(
            path='/',
            dirselect=True  # Permite selecionar pastas
        )

        # Layout principal do popup, contendo o seletor de pastas e os botões
        layout_popup = BoxLayout(orientation='vertical')
        layout_popup.add_widget(self.seletor_pasta)

        # Layout horizontal para os botões Confirmar e Fechar
        layout_botoes = BoxLayout(
            size_hint=(1, 0.1),
            spacing=10
        )

        # Botão Confirmar
        botao_confirmar = Button(text="Confirmar")
        botao_confirmar.bind(on_release=self.confirmar_selecao)
        layout_botoes.add_widget(botao_confirmar)

        # Botão Fechar
        botao_fechar = Button(text="Fechar")
        botao_fechar.bind(on_release=lambda x: self.janela_popup.dismiss())
        layout_botoes.add_widget(botao_fechar)

        # Adiciona os botões ao layout principal do popup
        layout_popup.add_widget(layout_botoes)

        # Cria e exibe o popup
        self.janela_popup = Popup(
            title="Escolha uma pasta",
            content=layout_popup,
            size_hint=(0.9, 0.9)
        )
        self.janela_popup.open()

    # Função para confirmar a escolha da pasta
    def confirmar_selecao(self, instancia):
        # Obtém a pasta selecionada
        self.dados = self.seletor_pasta.selection

        if self.dados:
            # Ativa o botão para calcular os parâmetros
            self.botao_calcular_parametros.disabled = False
        else:
            print("Nenhuma pasta selecionada.")

        # Fecha o popup
        self.janela_popup.dismiss()

    # Função para exibir um novo layout com os resultados. Ao clicar, também realiza os cálculos

    def mostrar_novo_layout(self, instancia):
        # Cria um novo layout flutuante
        novo_layout = FloatLayout()

        # Adiciona a imagem de fundo
        imagem_fundo = Image(
            source='Engrenagens.png',
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1)
        )
        novo_layout.add_widget(imagem_fundo)

        # Parâmetros dos botões
        largura_botao = 250
        altura_botao = 0.1
        espacamento_botoes = 0.05

        # Botão: Frequências Naturais / Fatores de Amortecimento
        botao_frequencias = Button(
            text="Frequências Naturais / Fatores de Amortecimento",
            size_hint=(0.3, altura_botao),
            width=largura_botao,
            pos_hint={'x': 0, 'top': 0.75}
        )
        botao_frequencias.bind(on_release=self.parametros)

        # Botão: Exibir FRF
        botao_frf = Button(
            text="Exibir FRF",
            size_hint=(0.3, altura_botao),
            width=largura_botao,
            pos_hint={'x': 0, 'top': 0.75 - (altura_botao + espacamento_botoes)}
        )
        botao_frf.bind(on_release=self.grafico)

        # Botão: MAC
        botao_mac = Button(
            text="MAC",
            size_hint=(0.3, altura_botao),
            width=largura_botao,
            pos_hint={'x': 0, 'top': 0.75 - 2 * (altura_botao + espacamento_botoes)}
        )
        botao_mac.bind(on_release=self.mac)

        # Botão: Exibir Modos de Vibração
        botao_modos = Button(
            text="Exibir Modos de Vibração",
            size_hint=(0.3, altura_botao),
            width=largura_botao,
            pos_hint={'x': 0, 'top': 0.75 - 3 * (altura_botao + espacamento_botoes)}
        )
        botao_modos.bind(on_press=self.mostrar_formas_modais)

        # Botão: Voltar ao layout anterior
        botao_voltar = Button(
            text="Voltar",
            size_hint=(None, altura_botao),
            width=largura_botao,
            pos_hint={'x': 0, 'bottom': 0.05}
        )
        botao_voltar.bind(on_press=self.go_back)

        # Adiciona todos os botões ao novo layout
        novo_layout.add_widget(botao_frequencias)
        novo_layout.add_widget(botao_frf)
        novo_layout.add_widget(botao_mac)
        novo_layout.add_widget(botao_modos)
        novo_layout.add_widget(botao_voltar)

        # Substitui o layout atual pelo novo layout
        self.root.clear_widgets()
        self.root.add_widget(novo_layout)

        # Guarda o caminho da pasta selecionada
        caminho_pasta = self.dados[0]
        self.FRF = []

        def extract_number(filename):
            # Encontra números inteiros ou decimais no nome do arquivo
            match = re.search(r'\d+\.\d+|\d+', filename)
            return float(match.group()) if match else float('inf')

        if os.path.isdir(caminho_pasta):
            # Ordena os arquivos numericamente com base nos números nos nomes, ignorando ".DS_Store"
            arquivos_ordenados = sorted(
                [arquivo for arquivo in os.listdir(caminho_pasta)
                 if os.path.isfile(os.path.join(caminho_pasta, arquivo)) and arquivo != ".DS_Store"],
                key=extract_number
            )

            # Itera sobre cada arquivo ordenado
            for arquivo in arquivos_ordenados:
                caminho_completo = os.path.join(caminho_pasta, arquivo)
                print(caminho_completo)
                # Carrega o arquivo UFF e extrai os dados FRF e frequência
                uff_arquivo = pyuff.UFF(caminho_completo)
                print("Arquivo carregado:", os.path.basename(caminho_completo))
                dados = uff_arquivo.read_sets()

                # Extrai a FRF e adiciona à lista
                self.y = dados["data"]
                print()
                print("VALOR FRF")
                print()
                print(self.y)
                self.freq = dados["x"]

                self.FRF.append(self.y)

        else:
            print("O caminho especificado não é uma pasta válida.")

        # Função para perguntar o valor da banda de frequência que se deseja trabalhar

        def confirmar_valor(instance):
            try:
                # Tenta acessar self.freq, se não houver o atributo, será lançado o AttributeError
                banda = float(text_input.text)  # Aceitar valores decimais
                if 0 < banda <= self.freq[-1]:
                    popup.dismiss()
                    processar_banda(banda)
                else:
                    mensagem.text = f"Insira um valor entre 0 e {self.freq[-1]:.2f},Recomenda-se trabalhar com uma banda até próxima do terceiro modo."
            except ValueError:
                # Erro quando o valor inserido não for um número
                mensagem.text = "Por favor, insira um número válido."
            except AttributeError:
                # Caso o atributo 'freq' não exista
                mensagem.text = (
                    "Erro ao escolher a pasta. Por favor, verifique se na pasta existem "
                    "somente arquivos UFF ou se o cabeçalho dos arquivos está correto."
                )

        # Função para processar a banda de frequência
        def processar_banda(banda):
            self.MODELO = pyEMA.Model(
                frf=self.FRF,
                freq=self.freq,
                lower=20,  # Frequência inferior fixa
                upper=banda,  # Frequência superior fornecida pelo usuário
                pol_order_high=40,
            )

            self.MODELO.get_poles()
            # Selecionar polos próximos às frequências naturais fornecidas

            #self.MODELO.select_poles()

            n_freq = [44.06, 175.94, 386.64, 681]

            self.MODELO.select_closest_poles(n_freq)

            # Calcular constantes modais e salvar

            self.frf_rec, self.modal_const = self.MODELO.get_constants(
                whose_poles="own", FRF_ind="all"
            )
            # Exibir frequências naturais e fatores de amortecimento
            print("Frequências Naturais:", self.MODELO.nat_freq)
            print("Fatores de Amortecimento:", self.MODELO.nat_xi)
            print(self.MODELO.normal_mode())
            # Gerar o gráfico

        # Layout do popup
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        text_input = TextInput(hint_text="Digite a banda de frequência (Hz)", multiline=False, input_filter='float')
        mensagem = Label(text=f"Insira um valor entre 0 e {self.freq[-1]:.2f} Hz. Recomensa-se trabalhar com uma banda até próxima do terceiro modo")
        confirmar_button = Button(text="Confirmar", size_hint_y=None, height=40)
        confirmar_button.bind(on_release=confirmar_valor)
        fechar_button = Button(text="Fechar", size_hint_y=None, height=40)
        fechar_button.bind(on_release=lambda x: popup.dismiss())

        # Adiciona widgets ao layout
        layout.add_widget(mensagem)
        layout.add_widget(text_input)
        layout.add_widget(confirmar_button)
        layout.add_widget(fechar_button)

        # Cria e exibe o popup
        popup = Popup(title="Selecione a Banda de Frequência", content=layout, size_hint=(0.7, 0.4))
        popup.open()

    # Função para mostrar os resultados já cálculados das frequências naturais e fatores de amortecimento

    def parametros(self, instance):
        print("parâmetros")
        print(self.MODELO.nat_freq)
        print(self.MODELO.nat_xi)

        layout = GridLayout(cols=2, spacing=10, padding=10)
        layout.add_widget(Label(text="Frequência Natural (Hz)", bold=True))
        layout.add_widget(Label(text="Amortecimento ξ", bold=True))

        for freq, xi in zip(self.MODELO.nat_freq, self.MODELO.nat_xi):
            layout.add_widget(TextInput(text=f"{freq:.2f}", readonly=True, halign="center", size_hint_y=None, height=40))
            layout.add_widget(TextInput(text=f"{xi:.5f}", readonly=True, halign="center", size_hint_y=None, height=40))

        def salvar_resultados(instance):
            # Criar o layout do popup de salvamento
            content = BoxLayout(orientation='vertical', spacing=10, padding=10)

            # FileChooser + TextInput para nome do arquivo
            filechooser = FileChooserIconView(path='.', filters=["*.txt"], size_hint=(1, 0.8))

            filename_input = TextInput(hint_text="Digite o nome do arquivo.txt", size_hint=(1, 0.1))

            buttons = BoxLayout(size_hint=(1, 0.1))
            btn_salvar = Button(text="Salvar")
            btn_cancelar = Button(text="Cancelar")

            buttons.add_widget(btn_salvar)
            buttons.add_widget(btn_cancelar)

            content.add_widget(filechooser)
            content.add_widget(filename_input)
            content.add_widget(buttons)

            popup_salvar = Popup(title="Salvar Arquivo", content=content, size_hint=(0.9, 0.9))

            def confirmar_salvamento(btn):
                dirpath = filechooser.path
                filename = filename_input.text.strip()

                if not filename.endswith(".txt"):
                    filename += ".txt"

                full_path = os.path.join(dirpath, filename)

                try:
                    with open(full_path, "w") as f:
                        f.write("Frequência Natural (Hz)\tAmortecimento (%)\n")
                        for freq, xi in zip(self.MODELO.nat_freq, self.MODELO.nat_xi):
                            f.write(f"{freq:.2f}\t{xi:.5f}\n")
                    print(f"Arquivo salvo em: {full_path}")
                    popup_salvar.dismiss()
                except Exception as e:
                    print("Erro ao salvar arquivo:", e)

            btn_salvar.bind(on_release=confirmar_salvamento)
            btn_cancelar.bind(on_release=popup_salvar.dismiss)

            popup_salvar.open()

        save_button = Button(text="Salvar", size_hint_y=None, height=40)
        save_button.bind(on_release=salvar_resultados)

        close_button = Button(text="Fechar", size_hint_y=None, height=40)
        close_button.bind(on_release=lambda x: popup.dismiss())

        buttons_layout = BoxLayout(size_hint_y=None, height=40)
        buttons_layout.add_widget(save_button)
        buttons_layout.add_widget(close_button)

        layout.add_widget(buttons_layout)

        popup = Popup(title="Resultados", content=layout, size_hint=(0.7, 0.7))
        popup.open()

# Função para mostrar a FRF reconstruída e a experimental

    def grafico(self, instance):
        # Função para processar o valor do popup
        def confirmar_valor(instance):
            try:
                valor = int(text_input.text)
                if 0 <= valor < len(self.FRF):
                    popup.dismiss()
                    desenhar_grafico(valor)
                else:
                    mensagem.text = f"Insira um valor entre 0 e {len(self.FRF) - 1}."
            except ValueError:
                mensagem.text = "Por favor, insira um número válido."

        # Função para desenhar o gráfico
        def desenhar_grafico(select_loc):
            plt.close('all')
            freq_a = self.MODELO.freq

            plt.figure(figsize=(10, 6))
            # Gráfico superior
            plt.subplot(211)
            plt.semilogy(self.freq, np.abs(self.FRF[select_loc]), label='Experiment')
            plt.semilogy(freq_a, np.abs(self.frf_rec[select_loc]), '--', label='LSCF')
            plt.xlim(0, self.freq[-1])
            plt.ylabel(r"MAGNITUDE")
            plt.legend(loc='best')

            # Gráfico inferior
            plt.subplot(212)
            plt.plot(self.freq, np.angle(self.FRF[select_loc], deg=1), label='Experiment')
            plt.plot(freq_a, np.angle(self.frf_rec[select_loc], deg=1), '--', label='LSCF')
            plt.xlim(0, self.freq[-1])
            plt.ylabel(r"FASE")
            plt.legend(loc='best')
            plt.show()

        # Layout do popup
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        text_input = TextInput(hint_text="Digite um valor", multiline=False, input_filter='int')
        mensagem = Label(text=f"Insira um valor entre 0 e {len(self.FRF) - 1}.")
        confirmar_button = Button(text="Confirmar", size_hint_y=None, height=40)
        confirmar_button.bind(on_release=confirmar_valor)
        fechar_button = Button(text="Fechar", size_hint_y=None, height=40)
        fechar_button.bind(on_release=lambda x: popup.dismiss())

        # Adiciona widgets ao layout
        layout.add_widget(mensagem)
        layout.add_widget(text_input)
        layout.add_widget(confirmar_button)
        layout.add_widget(fechar_button)

        # Cria e exibe o popup
        popup = Popup(title="Selecione o índice", content=layout, size_hint=(0.6, 0.4))
        popup.open()

# Função para mostrar o MAC

    def mac(self, instance):
        def calcular_MAC(matrix1, matrix2):
            n_modos_1 = matrix1.shape[1]
            n_modos_2 = matrix2.shape[1]
            MAC = np.zeros((n_modos_1, n_modos_2))

            for i in range(n_modos_1):
                for j in range(n_modos_2):
                    numerador = np.abs(np.dot(matrix1[:, i].conj().T, matrix2[:, j])) ** 2
                    denominador = (np.dot(matrix1[:, i].conj().T, matrix1[:, i]) *
                                   np.dot(matrix2[:, j].conj().T, matrix2[:, j]))
                    MAC[i, j] = numerador / denominador
            return MAC

        # Função para processar a matriz inserida pelo usuário
        def processar_input_matriz(input_text):
            try:
                matriz_usuario = eval(input_text)  # Use com cautela
                matriz_usuario = np.array(matriz_usuario, dtype=np.complex128)
                matriz_modelo = self.MODELO.A

                if matriz_modelo.shape[0] != matriz_usuario.shape[0]:
                    print("Erro: As matrizes devem ter o mesmo número de linhas (graus de liberdade).")
                    return

                mac_matrix = calcular_MAC(matriz_modelo, matriz_usuario)

                # Plotar a matriz MAC
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')

                # Coordenadas
                xpos, ypos = np.meshgrid(np.arange(mac_matrix.shape[0]), np.arange(mac_matrix.shape[1]), indexing="ij")
                xpos = xpos.flatten()
                ypos = ypos.flatten()
                zpos = np.zeros_like(xpos)

                # Dimensões das barras
                dx = dy = 0.5 * np.ones_like(zpos)
                dz = mac_matrix.flatten()

                # Cores
                colors = plt.cm.coolwarm(dz)  # você pode trocar o colormap

                # Desenhar barras
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')

                # Eixos e rótulos
                ax.set_xlabel('Modos do Modelo Experimental')
                ax.set_ylabel('Modos Inseridos')
                ax.set_zlabel('MAC')
                ax.set_xticks(np.arange(mac_matrix.shape[0]))
                ax.set_yticks(np.arange(mac_matrix.shape[1]))
                ax.set_xticklabels([f"Modo {i + 1}" for i in range(mac_matrix.shape[0])])
                ax.set_yticklabels([f"Modo {i + 1}" for i in range(mac_matrix.shape[1])])
                ax.set_zlim(0, 1)

                plt.title("Matriz MAC - Gráfico 3D", fontsize=16)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print("Erro ao processar a matriz:", e)

        # Layout do Popup
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        layout.add_widget(Label(text="Insira a matriz para comparação (formato Python):", size_hint_y=None, height=30))

        input_matriz = TextInput(
            multiline=True,
            font_size=14,
            size_hint=(1, 0.6),
            hint_text='Exemplo de matriz:\n[\n    [ 0.08813073,  0.14915783, -0.1929955 ],\n    [ 0.08813073,  0.14915783, -0.1929955 ],\n    [ 0.08813073,  0.14915783, -0.1929955 ],\n    [ 0.40585534,  0.49358053,  0.02056268],\n    [ 0.48601029,  0.1914442 ,  0.40785238],\n    [ 0.52895579, -0.19167906,  0.43849263],\n    [ 0.48265329, -0.49051556,  0.07745023],\n    [ 0.17486429, -0.42437681, -0.48639819],\n    [ 0.17242791, -0.41699746, -0.48321047],\n    [ 0.05212521, -0.1471539 , -0.23052535]\n]'
        )

        botoes = BoxLayout(size_hint=(1, 0.2), spacing=10)
        btn_confirmar = Button(text="Comparar")
        btn_cancelar = Button(text="Cancelar")

        botoes.add_widget(btn_confirmar)
        botoes.add_widget(btn_cancelar)

        layout.add_widget(input_matriz)
        layout.add_widget(botoes)

        popup = Popup(title="Inserir Matriz para MAC", content=layout, size_hint=(0.9, 0.9))

        btn_confirmar.bind(on_release=lambda x: [processar_input_matriz(input_matriz.text), popup.dismiss()])
        btn_cancelar.bind(on_release=popup.dismiss)

        popup.open()

# Função para mostrar as formas modais

    def mostrar_formas_modais(self, instance):
        plt.close('all')
        print("Exibindo formas modais...")

        formas_modais = self.MODELO.normal_mode()
        formas_modais_complexas = self.MODELO.A * 10  # <- MOVIDO PARA CÁ
        x_original = np.arange(formas_modais.shape[0])
        x_interp = np.linspace(0, len(x_original) - 1, 300)

        def seno(x, A, w, phi, offset):
            return A * np.sin(w * x + phi) + offset

        def animar_modo(modo):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Animação da Forma Modal {modo + 1}")
            ax.set_xlabel("Posição na Viga")
            ax.set_ylabel("Amplitude de Vibração")
            ax.set_xlim(0, len(x_original) - 1)
            ax.set_ylim(-3, 3)

            y = formas_modais[:, modo]

            # Interpolação suave usando interp1d
            interpolacao = interp1d(x_original, y, kind='cubic')
            y_interp = interpolacao(x_interp)

            linha, = ax.plot(x_interp, y_interp, color='blue', label=f"Modo {modo + 1}")
            pontos = ax.scatter(x_interp, y_interp, color='blue', s=10)

            def atualizar(frame):
                t = frame / 30
                y_animado = y_interp * np.sin(2 * np.pi * t)
                linha.set_ydata(y_animado)
                pontos.set_offsets(np.c_[x_interp, y_animado])
                return linha, pontos

            animacao = FuncAnimation(fig, atualizar, frames=300, interval=50, blit=False)
            ax.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        for modo in range(formas_modais.shape[1]):
            animar_modo(modo)

        # Layout para exibição das matrizes
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Título das matrizes
        layout.add_widget(Label(text="Formas Modais (Reais)", bold=True, size_hint_y=None, height=30))

        scroll_real = ScrollView(size_hint=(1, 0.4))
        grid_real = GridLayout(cols=1, size_hint_y=None)
        grid_real.bind(minimum_height=grid_real.setter('height'))

        for linha in formas_modais:
            texto = '\t'.join(f"{val:.6f}" for val in linha)
            grid_real.add_widget(TextInput(text=texto, readonly=True, size_hint_y=None, height=30, font_size=12))

        scroll_real.add_widget(grid_real)
        layout.add_widget(scroll_real)

        layout.add_widget(Label(text="Formas Modais (Complexas)", bold=True, size_hint_y=None, height=30))

        scroll_complexa = ScrollView(size_hint=(1, 0.4))
        grid_complexa = GridLayout(cols=1, size_hint_y=None)
        grid_complexa.bind(minimum_height=grid_complexa.setter('height'))

        for linha in formas_modais_complexas:
            texto = '\t'.join(f"{val.real:.6f}+{val.imag:.6f}j" for val in linha)
            grid_complexa.add_widget(TextInput(text=texto, readonly=True, size_hint_y=None, height=30, font_size=12))

        scroll_complexa.add_widget(grid_complexa)
        layout.add_widget(scroll_complexa)

        # Função para salvar as matrizes
        def salvar_formas_modais(instance):
            content = BoxLayout(orientation='vertical', spacing=10, padding=10)

            filechooser = FileChooserIconView(path='.', filters=["*.txt"], size_hint=(1, 0.8))
            filename_input = TextInput(hint_text="Digite o nome do arquivo.txt", size_hint=(1, 0.1))

            buttons = BoxLayout(size_hint=(1, 0.1))
            btn_salvar = Button(text="Salvar")
            btn_cancelar = Button(text="Cancelar")
            buttons.add_widget(btn_salvar)
            buttons.add_widget(btn_cancelar)

            content.add_widget(filechooser)
            content.add_widget(filename_input)
            content.add_widget(buttons)

            popup_salvar = Popup(title="Salvar Formas Modais", content=content, size_hint=(0.9, 0.9))

            def confirmar_salvamento(btn):
                dirpath = filechooser.path
                filename = filename_input.text.strip()

                if not filename.endswith(".txt"):
                    filename += ".txt"

                full_path = os.path.join(dirpath, filename)

                try:
                    with open(full_path, "w") as f:
                        f.write("Matriz das Formas Modais (Reais):\n")
                        for linha in formas_modais:
                            f.write('\t'.join(f"{val:.6f}" for val in linha) + '\n')

                        f.write("\nMatriz das Formas Modais (Complexas):\n")
                        for linha in formas_modais_complexas:
                            f.write('\t'.join(f"{val.real:.6f}+{val.imag:.6f}j" for val in linha) + '\n')

                    print(f"Formas modais salvas em: {full_path}")
                    popup_salvar.dismiss()
                except Exception as e:
                    print("Erro ao salvar arquivo:", e)

            btn_salvar.bind(on_release=confirmar_salvamento)
            btn_cancelar.bind(on_release=popup_salvar.dismiss)

            popup_salvar.open()

        # Botões de ação no fim do popup
        botoes_finais = BoxLayout(size_hint_y=None, height=40)
        btn_salvar = Button(text="Salvar")
        btn_salvar.bind(on_release=salvar_formas_modais)

        btn_fechar = Button(text="Fechar")
        btn_fechar.bind(on_release=lambda x: popup.dismiss())

        botoes_finais.add_widget(btn_salvar)
        botoes_finais.add_widget(btn_fechar)
        layout.add_widget(botoes_finais)

        popup = Popup(title="Formas Modais", content=layout, size_hint=(0.9, 0.9))
        popup.open()

    def go_back(self, instance):
        # Para o aplicativo atual
        self.stop()
        # Reinicia o aplicativo criando uma nova instância
        AME().run()

# Função para mostrar uma mensagem de tutorial

    def exibir_mensagem(self, instance):

        # texto do tutorial
        texto = (
            "Tutorial – Como usar o aplicativo de Análise Modal Experimental\n"
            "1. Entenda o objetivo:\n"
            "A análise modal experimental identifica as propriedades dinâmicas de uma estrutura, como:\n"
            "- Frequências naturais\n"
            "- Modos de vibração\n"
            "- Fatores de amortecimento\n"
            "Esses parâmetros são extraídos a partir de dados reais medidos em ensaios com resposta em frequência (FRF).\n\n"

            "2. Selecione o arquivo de entrada:\n"
            "Clique no botão para inserir o arquivo de FRF medida.\n"
            "O arquivo deve estar no formato .uff.\n\n"

            "3. Calcule os parâmetros:\n"
            "Clique no botão \"Calcular Parâmetros\" para iniciar a análise.\n"
            "O aplicativo processará os dados e abrirá a próxima etapa.\n\n"

            "4. Escolha a banda de frequência:\n"
            "Uma janela irá solicitar a faixa de frequência a ser analisada.\n"
            "Recomenda-se escolher uma faixa que inclua até o terceiro modo de vibração.\n\n"

            "5. Visualize o gráfico de estabilidade:\n"
            "Será exibido um gráfico com os polos estimados.\n"
            "Selecione os polos estáveis que representam os modos reais da estrutura.\n\n"

            "6. Acesse os resultados:\n"
            "Após a seleção dos polos, estarão disponíveis os botões para:\n"
            "- Visualizar os resultados (modos de vibração, FRF reconstruída, frequências naturais e amortecimento).\n"
            "- Salvar os dados da forma que desejar."
        )

        # Layout principal do Popup
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Título
        title_label = Label(
            text="Definição de Análise Modal",
            size_hint_y=None,
            height=30,
            font_size='16sp',
            bold=True,
            halign='center',
            valign='middle'
        )

        # Caixa de rolagem para o texto
        scrollview = ScrollView(size_hint=(1, 1))
        text_label = Label(
            text=texto,
            size_hint_y=None,
            text_size=(480, None),
            halign='left',
            valign='top',
            font_size='14sp'
        )
        text_label.bind(texture_size=text_label.setter('size'))  # Ajusta o tamanho da label ao conteúdo
        scrollview.add_widget(text_label)

        # Botão de fechar
        close_button = Button(
            text="Fechar",
            size_hint=(None, None),
            size=(100, 40),
            pos_hint={'center_x': 0.5}
        )
        close_button.bind(on_release=lambda x: popup.dismiss())

        # Adiciona tudo ao layout do popup
        popup_layout.add_widget(title_label)
        popup_layout.add_widget(scrollview)
        popup_layout.add_widget(close_button)

        # Cria o Popup
        popup = Popup(
            title='',
            content=popup_layout,
            size_hint=(None, None),
            size=(580, 400)
        )

        # Exibe o Popup
        popup.open()

# Faz rodar o aplicativo

AME().run()