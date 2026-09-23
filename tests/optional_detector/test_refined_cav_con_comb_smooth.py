"""Run: python3 -m unittest discover -s tests -p 'test_refined_cav_con_comb_smooth.py' -v."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import refined_cav_con_comb_smooth as cavity


def gridded_surface(z_function, extent=5, n=50):
    x,y=np.meshgrid(np.linspace(-extent,extent,n+1),np.linspace(-extent,extent,n+1))
    p=np.stack([x,y,z_function(x,y)],axis=-1)
    return np.array([tri for i in range(n) for j in range(n)
                     for tri in [[p[i,j],p[i+1,j],p[i,j+1]],
                                 [p[i+1,j],p[i+1,j+1],p[i,j+1]]]])


class SurfaceDetectionTests(unittest.TestCase):
    def test_graph_does_not_join_overlapping_disconnected_surfaces(self):
        a=gridded_surface(lambda x,y:x*0+1,n=5)
        b=a+np.array([0,0,.05])
        g=cavity.mesh_geometry(np.concatenate([a,b]))
        components=cavity.surface_components(g,np.ones(len(g['vectors']),bool),min_area=.1)
        self.assertEqual(len(components),2)

    def test_duplicate_faces_do_not_change_physical_area(self):
        a=gridded_surface(lambda x,y:x*0+1,n=5)
        g=cavity.mesh_geometry(np.concatenate([a,a[:,::-1]]))
        self.assertEqual(len(g['vectors']),len(a))
        self.assertAlmostEqual(g['area'].sum(),100)

    def test_reference_rejects_a_synthetic_depressed_floor(self):
        top=gridded_surface(lambda x,y:np.where((abs(x)<1.2)&(abs(y)<2),1.,3.))
        base=gridded_surface(lambda x,y:x*0,n=10)
        g=cavity.mesh_geometry(np.concatenate([top,base]))
        candidates,reference,fields=cavity.discover_candidates(g)
        self.assertTrue(candidates)
        center_height=cavity.measurement.quadric_z(np.array([0.]),np.array([0.]),reference)[0]
        self.assertAlmostEqual(center_height,3,delta=.12)
        best=candidates[0]
        self.assertLess(np.linalg.norm(best['center'][:2]),.3)
        self.assertAlmostEqual(best['depth_median_mm'],2,delta=.12)

    def test_proximal_floor_can_be_shallower_than_occlusal(self):
        g={'centers':np.array([[0,0,0],[1,0,1]])}
        o={'ids':np.array([0]),'score':10,'area_mm2':2,'depth_median_mm':2.5,'edge_distance_mm':3,'concave_fraction':1}
        p={'ids':np.array([1]),'score':2,'area_mm2':1,'depth_median_mm':1.2,'edge_distance_mm':.5,'concave_fraction':1}
        selected_o,selected_p=cavity.choose_candidates([o,p],g)
        self.assertIs(selected_o,o);self.assertIs(selected_p,p)

    def test_convex_crown_skirt_is_not_a_proximal_floor(self):
        g={'centers':np.array([[0,0,0],[1,0,1]])}
        o={'ids':np.array([0]),'score':10,'area_mm2':2,'depth_median_mm':2.5,'edge_distance_mm':3,'concave_fraction':1}
        skirt={'ids':np.array([1]),'score':2,'area_mm2':8,'depth_median_mm':1.2,'edge_distance_mm':.5,'concave_fraction':.1}
        self.assertIsNone(cavity.choose_candidates([o,skirt],g)[1])

    def test_nearby_but_disconnected_depression_is_not_paired(self):
        g={'centers':np.array([[0,0,0],[1,0,0]]),'pairs':np.empty((0,2),int)}
        o={'ids':np.array([0]),'score':10,'area_mm2':2,'depth_median_mm':2.5,'edge_distance_mm':3,'concave_fraction':1}
        p={'ids':np.array([1]),'score':2,'area_mm2':1,'depth_median_mm':1.2,'edge_distance_mm':.5,'concave_fraction':1}
        fields={'crown':np.ones(2,bool),'depth':np.ones(2),'top_gap':np.zeros(2)}
        self.assertIsNone(cavity.choose_candidates([o,p],g,fields)[1])

    def test_surface_path_preserves_a_shallower_connected_proximal_floor(self):
        g={'centers':np.array([[0,0,0],[1,0,1]]),'pairs':np.array([[0,1]])}
        o={'ids':np.array([0]),'score':10,'area_mm2':2,'depth_median_mm':2.5,'edge_distance_mm':3,'concave_fraction':1}
        p={'ids':np.array([1]),'score':2,'area_mm2':1,'depth_median_mm':1.2,'edge_distance_mm':.5,'concave_fraction':1}
        fields={'crown':np.ones(2,bool),'depth':np.array([2.5,1.2]),'top_gap':np.zeros(2)}
        self.assertIs(cavity.choose_candidates([o,p],g,fields)[1],p)

    def test_constant_width_flat_open_floor_has_no_forced_boundary(self):
        x,y=np.meshgrid(np.linspace(-.5,.5,20),np.linspace(0,5,80))
        centers=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])
        g={'centers':centers,'area':np.full(len(centers),.01)}
        candidate={'ids':np.arange(len(centers))}
        fields={'edge':centers[:,1]}
        self.assertIsNone(cavity.partition_opening(candidate,g,fields))

    def test_cavity_growth_remains_on_connected_surface(self):
        a=gridded_surface(lambda x,y:x*0+1,n=5)
        b=a+np.array([0,0,.05]);g=cavity.mesh_geometry(np.concatenate([a,b]))
        allowed=np.ones(len(g['vectors']),bool)
        groups=cavity.surface_components(g,allowed,min_area=.1)
        fields={'crown':allowed,'depth':np.ones(len(allowed)),'top_gap':np.zeros(len(allowed))}
        regions=cavity.grow_cavity_regions(g,fields,{'ids':groups[0][:2]},None)
        self.assertTrue(np.isin(regions[0],groups[0]).all())
        self.assertEqual(len(regions[1]),0)


class MaskRegistrationTests(unittest.TestCase):
    def test_black_padding_is_excluded_from_tooth_silhouette(self):
        image=np.full((80,80,3),255,np.uint8)
        image[:15,:15]=0;image[25:65,30:60]=140
        mask=cavity.image_tooth_mask(Image.fromarray(image))
        self.assertEqual(mask.sum(),40*30)
        self.assertFalse(mask[:15,:15].any())

    def test_orthographic_projection_uses_original_world_coordinates(self):
        camera={'matrix':[[2,0,0,10],[0,-3,0,20]],'toward_camera':[0,0,1]}
        uv,depth=cavity.project(np.array([[1,2,3],[-1,1,5]]),camera)
        np.testing.assert_array_equal(uv,[[12,14],[8,17]])
        np.testing.assert_array_equal(depth,[3,5])

    def test_perspective_projection_rejects_points_behind_camera(self):
        camera={'matrix':[[10,0,5,0],[0,10,5,0],[0,0,1,0]]}
        uv,_=cavity.project(np.array([[1,2,2],[0,0,-1]]),camera)
        np.testing.assert_allclose(uv[0],[10,15])
        self.assertTrue((uv[1]<0).all())

    def test_mask_projection_does_not_label_hidden_back_surface(self):
        front=gridded_surface(lambda x,y:x*0+1,extent=1,n=10)
        back=front-np.array([0,0,1]);g=cavity.mesh_geometry(np.concatenate([front,back]))
        camera={'matrix':[[5,0,0,8],[0,5,0,8]],'toward_camera':[0,0,1]}
        support,visible=cavity.mask_face_support(g,camera,np.ones((16,16),bool))
        self.assertTrue(support[:len(front)].all())
        self.assertFalse(support[len(front):].any())

    def test_grid_mismatch_prevents_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'images';masks=root/'masks';images.mkdir();masks.mkdir()
            Image.new('RGB',(32,64),'gray').save(images/'O_1.png')
            Image.new('L',(64,64),255).save(masks/'O_1_mask.png')
            info,support,_,_=cavity.prepare_mask_guidance({},1,'Occlusal',images,masks)
            self.assertEqual(info['status'],'image_mask_grid_mismatch')
            self.assertFalse(info['use_for_selection']);self.assertIsNone(support)

    def test_ambiguous_mask_cannot_override_geometric_selection(self):
        g={'centers':np.array([[0,0,0],[1,0,1]]),'area':np.ones(2),'pairs':np.array([[0,1]])}
        o={'ids':np.array([0]),'score':10,'area_mm2':2,'depth_median_mm':2.5,'edge_distance_mm':3,'concave_fraction':1}
        p={'ids':np.array([1]),'score':2,'area_mm2':1,'depth_median_mm':1.2,'edge_distance_mm':.5,'concave_fraction':1}
        guidance={'Occlusal':({'use_for_selection':False},np.array([False,True]),None,None)}
        fields={'edge':np.array([3.,.5]),'depth':np.array([2.5,1.2]),'concavity':np.ones(2),
                'crown':np.ones(2,bool),'top_gap':np.zeros(2)}
        a,b,_=cavity.select_floors([o,p],g,fields,guidance)
        self.assertIs(a,o);self.assertIs(b,p)

    def test_good_silhouette_on_convex_surface_cannot_override_depression(self):
        g={'area':np.ones(2)}
        candidates=[{'ids':np.array([0]),'concave_fraction':1.,'edge_distance_mm':3.},
                    {'ids':np.array([1]),'concave_fraction':.1,'edge_distance_mm':1.}]
        info={'status':'estimated_usable_requires_review','use_for_selection':True}
        guidance={'Occlusal':(info,np.array([False,True]),None,None)}
        cavity.validate_estimated_mask_support(g,candidates,guidance)
        self.assertFalse(info['use_for_selection'])
        self.assertEqual(info['status'],'estimated_mask_geometry_disagreement')

    def test_consistent_mask_on_depression_remains_eligible(self):
        g={'area':np.ones(2)}
        candidates=[{'ids':np.array([0,1]),'concave_fraction':1.,'edge_distance_mm':3.}]
        info={'status':'estimated_usable_requires_review','use_for_selection':True}
        guidance={'Occlusal':(info,np.array([True,True]),None,None)}
        cavity.validate_estimated_mask_support(g,candidates,guidance)
        self.assertTrue(info['use_for_selection'])


if __name__=='__main__':unittest.main()
